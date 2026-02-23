/**
 * Local Sandbox Provider
 *
 * A sandbox implementation that executes commands on the local machine.
 * This is the default sandbox for development and local agents.
 *
 * Supports optional native OS sandboxing:
 * - macOS: Uses seatbelt (sandbox-exec) for filesystem and network isolation
 * - Linux: Uses bubblewrap (bwrap) for namespace isolation
 */

import * as childProcess from 'node:child_process';
import type { SpawnOptions } from 'node:child_process';
import * as crypto from 'node:crypto';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

import type { WorkspaceFilesystem } from '../filesystem/filesystem';
import type { FilesystemMountConfig, MountResult } from '../filesystem/mount';
import type { ProviderStatus } from '../lifecycle';
import { IsolationUnavailableError } from './errors';
import { MastraSandbox } from './mastra-sandbox';
import type { MastraSandboxOptions } from './mastra-sandbox';
import type { MountManager } from './mount-manager';
import { mountGCS } from './mounts/gcs';
import type { LocalGCSMountConfig } from './mounts/gcs';
import { isMountPoint, unmountFuse } from './mounts/platform';
import { mountS3 } from './mounts/s3';
import type { LocalS3MountConfig } from './mounts/s3';
import { MountToolNotFoundError } from './mounts/types';
import type { LocalMountContext } from './mounts/types';
import type { IsolationBackend, NativeSandboxConfig } from './native-sandbox';
import { detectIsolation, isIsolationAvailable, generateSeatbeltProfile, wrapCommand } from './native-sandbox';
import type { SandboxInfo, ExecuteCommandOptions, CommandResult } from './types';

/** Allowlist pattern for mount paths — absolute path with safe characters only. */
const SAFE_MOUNT_PATH = /^\/[a-zA-Z0-9_.\-/]+$/;

function validateMountPath(mountPath: string): void {
  if (!SAFE_MOUNT_PATH.test(mountPath)) {
    throw new Error(
      `Invalid mount path: ${mountPath}. Must be an absolute path with alphanumeric, dash, dot, underscore, or slash characters only.`,
    );
  }
}

interface ExecStreamingOptions extends Omit<SpawnOptions, 'timeout' | 'stdio'> {
  /** Timeout in ms - handled manually for custom exit code 124 */
  timeout?: number;
  onStdout?: (data: string) => void;
  onStderr?: (data: string) => void;
}

/**
 * Execute a command with optional streaming callbacks.
 * Uses spawn when callbacks are provided for real-time output.
 */
function execWithStreaming(
  command: string,
  args: string[],
  options: ExecStreamingOptions,
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const { timeout, onStdout, onStderr, cwd, env, ...spawnOptions } = options;
  return new Promise((resolve, reject) => {
    const proc = childProcess.spawn(command, args, { cwd, env, ...spawnOptions });

    let stdout = '';
    let stderr = '';
    let killed = false;

    // Set up timeout
    const timeoutId = timeout
      ? setTimeout(() => {
          killed = true;
          proc.kill('SIGTERM');
        }, timeout)
      : undefined;

    proc.stdout.on('data', (data: Buffer) => {
      const str = data.toString();
      stdout += str;
      onStdout?.(str);
    });

    proc.stderr.on('data', (data: Buffer) => {
      const str = data.toString();
      stderr += str;
      onStderr?.(str);
    });

    proc.on('error', err => {
      if (timeoutId) clearTimeout(timeoutId);
      const errorMsg = err.message;
      stderr += errorMsg;
      onStderr?.(errorMsg);
      reject(err);
    });

    proc.on('close', (code, signal) => {
      if (timeoutId) clearTimeout(timeoutId);
      if (killed) {
        const timeoutMsg = `\nProcess timed out after ${timeout}ms`;
        onStderr?.(timeoutMsg);
        resolve({ stdout, stderr: stderr + timeoutMsg, exitCode: 124 });
      } else if (signal) {
        // When terminated by signal, code is null but signal contains the signal name
        const signalMsg = `\nProcess terminated by ${signal}`;
        onStderr?.(signalMsg);
        resolve({ stdout, stderr: stderr + signalMsg, exitCode: 128 });
      } else {
        resolve({ stdout, stderr, exitCode: code ?? 0 });
      }
    });
  });
}

/**
 * Local sandbox provider configuration.
 */
export interface LocalSandboxOptions extends MastraSandboxOptions {
  /** Unique identifier for this sandbox instance */
  id?: string;
  /** Working directory for command execution */
  workingDirectory?: string;
  /**
   * Environment variables to set for command execution.
   * PATH is included by default unless overridden (needed for finding executables).
   * Other host environment variables are not inherited unless explicitly passed.
   *
   * @example
   * ```typescript
   * // Default - only PATH is available
   * env: undefined
   *
   * // Add specific variables
   * env: { NODE_ENV: 'production', HOME: process.env.HOME }
   *
   * // Full host environment (less secure)
   * env: process.env
   * ```
   */
  env?: NodeJS.ProcessEnv;
  /** Default timeout for operations in ms (default: 30000) */
  timeout?: number;
  /**
   * Isolation backend for sandboxed execution.
   * - 'none': No sandboxing (direct execution on host) - default
   * - 'seatbelt': macOS sandbox-exec (built-in on macOS)
   * - 'bwrap': Linux bubblewrap (requires installation)
   *
   * Use `LocalSandbox.detectIsolation()` to get the recommended backend.
   * @default 'none'
   */
  isolation?: IsolationBackend;
  /**
   * Configuration for native sandboxing.
   * Only used when isolation is 'seatbelt' or 'bwrap'.
   */
  nativeSandbox?: NativeSandboxConfig;
}

/**
 * Local sandbox implementation.
 *
 * Executes commands directly on the host machine.
 * This is the recommended sandbox for development and trusted local execution.
 *
 * @example
 * ```typescript
 * import { Workspace, LocalFilesystem, LocalSandbox } from '@mastra/core';
 *
 * const workspace = new Workspace({
 *   filesystem: new LocalFilesystem({ basePath: './my-workspace' }),
 *   sandbox: new LocalSandbox({ workingDirectory: './my-workspace' }),
 * });
 *
 * await workspace.init();
 * const result = await workspace.executeCommand('node', ['script.js']);
 * ```
 */
export class LocalSandbox extends MastraSandbox {
  readonly id: string;
  readonly name = 'LocalSandbox';
  readonly provider = 'local';

  status: ProviderStatus = 'pending';

  declare readonly mounts: MountManager;

  private readonly _workingDirectory: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly timeout?: number;
  private readonly _isolation: IsolationBackend;
  private _nativeSandboxConfig: NativeSandboxConfig;
  private _seatbeltProfile?: string;
  private _seatbeltProfilePath?: string;
  private _sandboxFolderPath?: string;
  private _userProvidedProfilePath = false;
  private readonly _createdAt: Date;
  private _activeMountPaths: Set<string> = new Set();

  /**
   * The working directory where commands are executed.
   */
  get workingDirectory(): string {
    return this._workingDirectory;
  }

  /**
   * The isolation backend being used.
   */
  get isolation(): IsolationBackend {
    return this._isolation;
  }

  /**
   * Detect the best available isolation backend for this platform.
   * Returns detection result with backend recommendation and availability.
   *
   * @example
   * ```typescript
   * const result = LocalSandbox.detectIsolation();
   * const sandbox = new LocalSandbox({
   *   isolation: result.available ? result.backend : 'none',
   * });
   * ```
   */
  static detectIsolation() {
    return detectIsolation();
  }

  constructor(options: LocalSandboxOptions = {}) {
    super({ ...options, name: 'LocalSandbox' });
    this.id = options.id ?? this.generateId();
    this._createdAt = new Date();
    // Default working directory is .sandbox/ in cwd - isolated from seatbelt profiles
    this._workingDirectory = options.workingDirectory ?? path.join(process.cwd(), '.sandbox');
    this.env = options.env ?? {};
    this.timeout = options.timeout;
    this._nativeSandboxConfig = options.nativeSandbox ?? {};

    // Validate and set isolation backend
    const requestedIsolation = options.isolation ?? 'none';
    if (requestedIsolation !== 'none' && !isIsolationAvailable(requestedIsolation)) {
      const detection = detectIsolation();
      throw new IsolationUnavailableError(requestedIsolation, detection.message);
    }
    this._isolation = requestedIsolation;
  }

  private generateId(): string {
    return `local-sandbox-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  /**
   * Build the environment object for execution.
   * Always includes PATH by default (needed for finding executables).
   * Merges the sandbox's configured env with any additional env from the command.
   */
  private buildEnv(additionalEnv?: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
    return {
      PATH: process.env.PATH, // Always include PATH for finding executables
      ...this.env,
      ...additionalEnv,
    };
  }

  /**
   * Start the local sandbox.
   * Creates working directory and sets up seatbelt profile if using macOS isolation.
   * Status management is handled by the base class.
   */
  async start(): Promise<void> {
    this.logger.debug('[LocalSandbox] Starting sandbox', {
      workingDirectory: this._workingDirectory,
      isolation: this._isolation,
    });

    await fs.mkdir(this.workingDirectory, { recursive: true });

    // Set up seatbelt profile for macOS sandboxing
    if (this._isolation === 'seatbelt') {
      const userProvidedPath = this._nativeSandboxConfig.seatbeltProfilePath;

      if (userProvidedPath) {
        // User provided a custom path
        this._seatbeltProfilePath = userProvidedPath;
        this._userProvidedProfilePath = true;

        // Check if file exists at user's path
        try {
          this._seatbeltProfile = await fs.readFile(userProvidedPath, 'utf-8');
        } catch (err: unknown) {
          if (err instanceof Error && 'code' in err && (err as NodeJS.ErrnoException).code !== 'ENOENT') {
            throw err;
          }
          // File doesn't exist, generate default and write to user's path
          this._seatbeltProfile = generateSeatbeltProfile(this.workingDirectory, this._nativeSandboxConfig);
          // Ensure parent directory exists
          await fs.mkdir(path.dirname(userProvidedPath), { recursive: true });
          await fs.writeFile(userProvidedPath, this._seatbeltProfile, 'utf-8');
        }
      } else {
        // No custom path, use default location
        this._seatbeltProfile = generateSeatbeltProfile(this.workingDirectory, this._nativeSandboxConfig);

        // Generate a deterministic hash from workspace path and config
        // This allows identical sandboxes to share profiles while preventing collisions
        const configHash = crypto
          .createHash('sha256')
          .update(this.workingDirectory)
          .update(JSON.stringify(this._nativeSandboxConfig))
          .digest('hex')
          .slice(0, 8);

        // Write profile to .sandbox-profiles/ in cwd (outside working directory)
        // This prevents sandboxed processes from reading/modifying their own security profile
        this._sandboxFolderPath = path.join(process.cwd(), '.sandbox-profiles');
        await fs.mkdir(this._sandboxFolderPath, { recursive: true });
        this._seatbeltProfilePath = path.join(this._sandboxFolderPath, `seatbelt-${configHash}.sb`);
        await fs.writeFile(this._seatbeltProfilePath, this._seatbeltProfile, 'utf-8');
      }
    }

    this.logger.debug('[LocalSandbox] Sandbox started', { workingDirectory: this._workingDirectory });
  }

  /**
   * Stop the local sandbox.
   * Unmounts all active FUSE mounts before stopping.
   * Status management is handled by the base class.
   */
  async stop(): Promise<void> {
    this.logger.debug('[LocalSandbox] Stopping sandbox', { workingDirectory: this._workingDirectory });

    // Unmount all active mounts (best-effort)
    for (const mountPath of [...this._activeMountPaths]) {
      try {
        await this.unmount(mountPath);
      } catch {
        // Best-effort unmount
      }
    }
  }

  /**
   * Destroy the local sandbox and clean up resources.
   * Unmounts all filesystems, clears mount state, and cleans up seatbelt profile.
   * Status management is handled by the base class.
   */
  async destroy(): Promise<void> {
    this.logger.debug('[LocalSandbox] Destroying sandbox', { workingDirectory: this._workingDirectory });

    // Unmount all active mounts
    for (const mountPath of [...this._activeMountPaths]) {
      try {
        await this.unmount(mountPath);
      } catch {
        // Ignore errors during cleanup
      }
    }
    this._activeMountPaths.clear();
    this.mounts.clear();

    // Clean up seatbelt profile only if it was auto-generated (not user-provided)
    if (this._seatbeltProfilePath && !this._userProvidedProfilePath) {
      try {
        await fs.unlink(this._seatbeltProfilePath);
      } catch {
        // Ignore errors if file doesn't exist
      }
    }
    this._seatbeltProfilePath = undefined;
    this._seatbeltProfile = undefined;
    this._userProvidedProfilePath = false;

    // Try to remove .sandbox folder if empty
    if (this._sandboxFolderPath) {
      try {
        await fs.rmdir(this._sandboxFolderPath);
      } catch {
        // Ignore errors - folder may not be empty or may not exist
      }
      this._sandboxFolderPath = undefined;
    }
  }

  async isReady(): Promise<boolean> {
    return this.status === 'running';
  }

  async getInfo(): Promise<SandboxInfo> {
    return {
      id: this.id,
      name: this.name,
      provider: this.provider,
      status: this.status,
      createdAt: this._createdAt,
      resources: {
        memoryMB: Math.round(os.totalmem() / 1024 / 1024),
        cpuCores: os.cpus().length,
      },
      metadata: {
        workingDirectory: this.workingDirectory,
        platform: os.platform(),
        nodeVersion: process.version,
        isolation: this._isolation,
        isolationConfig:
          this._isolation !== 'none'
            ? {
                allowNetwork: this._nativeSandboxConfig.allowNetwork ?? false,
                readOnlyPaths: this._nativeSandboxConfig.readOnlyPaths,
                readWritePaths: this._nativeSandboxConfig.readWritePaths,
              }
            : undefined,
      },
    };
  }

  getInstructions(): string {
    if (this.workingDirectory) {
      return `Local command execution. Working directory: "${this.workingDirectory}".`;
    }
    return 'Local command execution on the host machine.';
  }

  // ---------------------------------------------------------------------------
  // Mount Support
  // ---------------------------------------------------------------------------

  /**
   * Mount a filesystem at a path on the local host.
   * Uses FUSE tools (s3fs, gcsfuse) to mount cloud storage.
   */
  async mount(filesystem: WorkspaceFilesystem, mountPath: string): Promise<MountResult> {
    validateMountPath(mountPath);

    // Resolve virtual mount path to host filesystem path
    const hostPath = this.resolveHostPath(mountPath);

    this.logger.debug(`[LocalSandbox] Mounting "${mountPath}" → "${hostPath}"...`);

    // Get mount config
    const config = filesystem.getMountConfig?.() as FilesystemMountConfig | undefined;
    if (!config) {
      const error = `Filesystem "${filesystem.id}" does not provide a mount config`;
      this.logger.error(`[LocalSandbox] ${error}`);
      this.mounts.set(mountPath, { filesystem, state: 'error', error });
      return { success: false, mountPath, error };
    }

    // Check if already mounted with matching config
    const existingMount = await this.checkExistingMount(mountPath, hostPath, config);
    if (existingMount === 'matching') {
      this.logger.debug(
        `[LocalSandbox] Detected existing mount for ${filesystem.provider} ("${filesystem.id}") at "${hostPath}" with correct config, skipping`,
      );
      this.mounts.set(mountPath, { state: 'mounted', config });
      this._activeMountPaths.add(mountPath);
      return { success: true, mountPath };
    } else if (existingMount === 'mismatched') {
      this.logger.debug(`[LocalSandbox] Config mismatch, unmounting to re-mount with new config...`);
      await this.unmount(mountPath);
    }

    this.logger.debug(`[LocalSandbox] Config type: ${config.type}`);
    this.mounts.set(mountPath, { filesystem, state: 'mounting', config });

    // Check if host directory exists and is non-empty
    try {
      const entries = await fs.readdir(hostPath);
      if (entries.length > 0) {
        const error = `Cannot mount at ${hostPath}: directory exists and is not empty. Mounting would hide existing files. Use a different path or empty the directory first.`;
        this.logger.error(`[LocalSandbox] ${error}`);
        this.mounts.set(mountPath, { filesystem, state: 'error', config, error });
        return { success: false, mountPath, error };
      }
    } catch (err: unknown) {
      if (err instanceof Error && 'code' in err && (err as NodeJS.ErrnoException).code === 'ENOENT') {
        // Directory doesn't exist yet — will create it below
      } else if (err instanceof Error && 'code' in err && (err as NodeJS.ErrnoException).code !== 'ENOTDIR') {
        // Some other error (not ENOENT, not ENOTDIR) — proceed anyway
      }
    }

    // Create mount directory under working directory
    try {
      this.logger.debug(`[LocalSandbox] Creating mount directory at ${hostPath}...`);
      await fs.mkdir(hostPath, { recursive: true });
    } catch (mkdirError) {
      this.logger.debug(`[LocalSandbox] mkdir error for "${hostPath}":`, mkdirError);
      this.mounts.set(mountPath, { filesystem, state: 'error', config, error: String(mkdirError) });
      return { success: false, mountPath, error: String(mkdirError) };
    }

    // Create mount context
    const mountCtx = this.createMountContext();

    try {
      switch (config.type) {
        case 'local': {
          // Local filesystem — create a symlink from hostPath to the basePath
          const localConfig = config as { type: 'local'; basePath: string };
          // Remove the empty directory created above — symlink replaces it
          await fs.rmdir(hostPath);
          await fs.symlink(localConfig.basePath, hostPath);
          this.logger.debug(`[LocalSandbox] Symlinked local mount ${hostPath} → ${localConfig.basePath}`);
          break;
        }
        case 's3':
          this.logger.debug(`[LocalSandbox] Mounting S3 bucket at ${hostPath}...`);
          await mountS3(hostPath, config as LocalS3MountConfig, mountCtx);
          this.logger.debug(`[LocalSandbox] Mounted S3 bucket at ${hostPath}`);
          break;
        case 'gcs':
          this.logger.debug(`[LocalSandbox] Mounting GCS bucket at ${hostPath}...`);
          await mountGCS(hostPath, config as LocalGCSMountConfig, mountCtx);
          this.logger.debug(`[LocalSandbox] Mounted GCS bucket at ${hostPath}`);
          break;
        default:
          this.mounts.set(mountPath, {
            filesystem,
            state: 'unsupported',
            config,
            error: `Unsupported mount type: ${(config as FilesystemMountConfig).type}`,
          });
          return {
            success: false,
            mountPath,
            error: `Unsupported mount type: ${(config as FilesystemMountConfig).type}`,
          };
      }
    } catch (error) {
      // Tool not installed — warn and mark as unavailable (workspace still works via SDK)
      if (error instanceof MountToolNotFoundError) {
        this.logger.warn(
          `[LocalSandbox] FUSE mount unavailable at "${mountPath}": ${error.message}. Filesystem tools will still work, but sandbox processes won't have access to this mount path.`,
        );
        this.mounts.set(mountPath, { filesystem, state: 'unavailable', config, error: String(error) });

        try {
          await fs.rmdir(hostPath);
        } catch {
          // Ignore cleanup errors
        }

        return { success: false, mountPath, error: String(error), unavailable: true };
      }

      // Actual mount failure — error
      this.logger.error(
        `[LocalSandbox] Error mounting "${filesystem.provider}" (${filesystem.id}) at "${hostPath}":`,
        error,
      );
      this.mounts.set(mountPath, { filesystem, state: 'error', config, error: String(error) });

      // Clean up the directory we created since mount failed
      try {
        await fs.rmdir(hostPath);
        this.logger.debug(`[LocalSandbox] Cleaned up directory after failed mount: ${hostPath}`);
      } catch {
        // Ignore cleanup errors
      }

      return { success: false, mountPath, error: String(error) };
    }

    // Mark as mounted
    this.mounts.set(mountPath, { state: 'mounted', config });
    this._activeMountPaths.add(mountPath);

    // Write marker file
    await this.writeMarkerFile(mountPath, hostPath);

    // Dynamically add host path to isolation allowlist
    this.addMountPathToIsolation(hostPath);

    this.logger.debug(`[LocalSandbox] Mounted ${mountPath} → ${hostPath}`);
    return { success: true, mountPath };
  }

  /**
   * Unmount a filesystem from a path.
   */
  async unmount(mountPath: string): Promise<void> {
    validateMountPath(mountPath);

    const hostPath = this.resolveHostPath(mountPath);

    this.logger.debug(`[LocalSandbox] Unmounting ${mountPath} (${hostPath})...`);

    // Only call FUSE unmount for actual FUSE mounts (not symlinks)
    try {
      const stats = await fs.lstat(hostPath);
      if (!stats.isSymbolicLink()) {
        const mountCtx = this.createMountContext();
        await unmountFuse(hostPath, mountCtx);
      }
    } catch (error) {
      this.logger.debug(`[LocalSandbox] Unmount error:`, error);
    }

    this.mounts.delete(mountPath);
    this._activeMountPaths.delete(mountPath);

    // Clean up marker file
    const filename = this.mounts.markerFilename(hostPath);
    const markerPath = `/tmp/.mastra-mounts/${filename}`;
    try {
      await fs.unlink(markerPath);
    } catch {
      // Ignore if doesn't exist
    }

    // Remove mount point (symlink or empty directory)
    try {
      const stats = await fs.lstat(hostPath);
      if (stats.isSymbolicLink()) {
        await fs.unlink(hostPath);
      } else {
        await fs.rmdir(hostPath);
      }
      this.logger.debug(`[LocalSandbox] Unmounted and removed ${hostPath}`);
    } catch {
      this.logger.debug(`[LocalSandbox] Unmounted ${hostPath} (not removed: does not exist or not empty)`);
    }
  }

  // ---------------------------------------------------------------------------
  // Mount Helpers (private)
  // ---------------------------------------------------------------------------

  /**
   * Run a command on the host, outside any isolation.
   * Used for mount/unmount operations which need host-level access.
   */
  private runHostCommand(
    command: string,
    args: string[],
    options?: { timeout?: number },
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    return execWithStreaming(command, args, {
      cwd: this._workingDirectory,
      timeout: options?.timeout ?? 30_000,
      env: this.buildEnv(),
    });
  }

  /**
   * Create a LocalMountContext for mount operations.
   */
  private createMountContext(): LocalMountContext {
    return {
      run: (command, args, options) => this.runHostCommand(command, args, options),
      platform: os.platform(),
      logger: this.logger,
    };
  }

  /**
   * Write a marker file for detecting config changes.
   * Uses hostPath (resolved OS path) for the marker filename and content,
   * and mountPath (virtual path) for looking up the entry.
   */
  private async writeMarkerFile(mountPath: string, hostPath: string): Promise<void> {
    const entry = this.mounts.get(mountPath);
    if (!entry?.configHash) return;

    const filename = this.mounts.markerFilename(hostPath);
    const markerContent = `${hostPath}|${entry.configHash}`;
    const markerDir = '/tmp/.mastra-mounts';
    const markerFilePath = path.join(markerDir, filename);

    try {
      await fs.mkdir(markerDir, { recursive: true });
      await fs.writeFile(markerFilePath, markerContent, 'utf-8');
    } catch {
      this.logger.debug(`[LocalSandbox] Warning: Could not write marker file at ${markerFilePath}`);
    }
  }

  /**
   * Check if a path is already mounted and if the config matches.
   * Uses hostPath (resolved OS path) for checking the actual mount point.
   */
  private async checkExistingMount(
    _mountPath: string,
    hostPath: string,
    newConfig: FilesystemMountConfig,
  ): Promise<'not_mounted' | 'matching' | 'mismatched'> {
    const mountCtx = this.createMountContext();
    const mounted = await isMountPoint(hostPath, mountCtx);

    if (!mounted) {
      return 'not_mounted';
    }

    // Path is mounted — check if config matches via marker file
    const filename = this.mounts.markerFilename(hostPath);
    const markerPath = `/tmp/.mastra-mounts/${filename}`;

    try {
      const content = await fs.readFile(markerPath, 'utf-8');
      const parsed = this.mounts.parseMarkerContent(content.trim());

      if (!parsed) {
        return 'mismatched';
      }

      const newConfigHash = this.mounts.computeConfigHash(newConfig);
      this.logger.debug(
        `[LocalSandbox] Marker check — stored hash: "${parsed.configHash}", new config hash: "${newConfigHash}"`,
      );

      if (parsed.path === hostPath && parsed.configHash === newConfigHash) {
        return 'matching';
      }
    } catch {
      // Marker doesn't exist or can't be read
    }

    return 'mismatched';
  }

  /**
   * Dynamically add a mount path to the sandbox isolation allowlist.
   *
   * - Seatbelt: pushes to readWritePaths, regenerates inline profile
   * - Bwrap: pushes to readWritePaths (buildBwrapCommand reads config each call)
   */
  private addMountPathToIsolation(mountPath: string): void {
    if (this._isolation === 'none') return;

    // Add to readWritePaths
    if (!this._nativeSandboxConfig.readWritePaths) {
      this._nativeSandboxConfig = { ...this._nativeSandboxConfig, readWritePaths: [] };
    }
    if (!this._nativeSandboxConfig.readWritePaths!.includes(mountPath)) {
      this._nativeSandboxConfig.readWritePaths!.push(mountPath);
    }

    // Seatbelt: regenerate the inline profile so the next executeCommand() picks it up
    if (this._isolation === 'seatbelt') {
      this._seatbeltProfile = generateSeatbeltProfile(this.workingDirectory, this._nativeSandboxConfig);
    }
    // Bwrap: buildBwrapCommand reads config.readWritePaths each call, so no extra work needed
  }

  /**
   * Resolve a virtual mount path to a host filesystem path.
   * Virtual paths like "/s3" become "<workingDir>/s3".
   * E2B can use root-level paths via sudo, but LocalSandbox resolves under workingDirectory.
   */
  private resolveHostPath(mountPath: string): string {
    return path.join(this._workingDirectory, mountPath.replace(/^\/+/, ''));
  }

  /**
   * Wrap a command with the configured isolation backend.
   */
  private wrapCommandForIsolation(command: string, args: string[]): { command: string; args: string[] } {
    if (this._isolation === 'none') {
      return { command, args };
    }

    return wrapCommand(command, args, {
      backend: this._isolation,
      workspacePath: this.workingDirectory,
      seatbeltProfile: this._seatbeltProfile,
      config: this._nativeSandboxConfig,
    });
  }

  async executeCommand(
    command: string,
    args: string[] = [],
    options: ExecuteCommandOptions = {},
  ): Promise<CommandResult> {
    this.logger.debug('[LocalSandbox] Executing command', { command, args, cwd: options.cwd ?? this.workingDirectory });

    // Auto-start if not running (lazy initialization)
    await this.ensureRunning();

    const startTime = Date.now();

    // Wrap command with isolation backend if configured
    const wrapped = this.wrapCommandForIsolation(command, args);

    // Use streaming execution when callbacks are provided

    try {
      const result = await execWithStreaming(wrapped.command, wrapped.args, {
        cwd: options.cwd ?? this.workingDirectory,
        timeout: options.timeout ?? this.timeout ?? 30000,
        env: this.buildEnv(options.env),
        onStdout: options.onStdout,
        onStderr: options.onStderr,
      });

      const commandResult: CommandResult = {
        success: result.exitCode === 0,
        stdout: result.stdout,
        stderr: result.stderr,
        exitCode: result.exitCode,
        executionTimeMs: Date.now() - startTime,
      };

      this.logger.debug('[LocalSandbox] Command completed', {
        command,
        exitCode: commandResult.exitCode,
        executionTimeMs: commandResult.executionTimeMs,
      });

      return commandResult;
    } catch (error: unknown) {
      const executionTimeMs = Date.now() - startTime;
      this.logger.error('[LocalSandbox] Command failed', { command, error, executionTimeMs });
      return {
        success: false,
        stdout: '',
        stderr: error instanceof Error ? error.message : String(error),
        exitCode: 1,
        executionTimeMs,
      };
    }
  }
}
