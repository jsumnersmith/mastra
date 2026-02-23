---
'@mastra/core': minor
---

Added FUSE mount support to LocalSandbox for mounting cloud filesystems (S3, GCS) as local directories. When using LocalSandbox with remote filesystems like S3Filesystem or GCSFilesystem, the sandbox now mounts them via s3fs-fuse or gcsfuse so that spawned processes can access cloud storage through the local filesystem. Mount paths are automatically added to the sandbox isolation allowlist (seatbelt/bwrap). All mounts are cleaned up on stop/destroy.

**Usage example:**

```typescript
import { Workspace, LocalSandbox } from '@mastra/core/workspace';
import { S3Filesystem } from '@mastra/s3';

const workspace = new Workspace({
  mounts: {
    '/data': new S3Filesystem({ bucket: 'my-bucket', region: 'us-east-1' }),
  },
  sandbox: new LocalSandbox({ workingDirectory: './workspace' }),
});

await workspace.init();
// Spawned processes can now read/write /data
const result = await workspace.executeCommand('ls', ['/data']);
```

**Requirements:** s3fs-fuse (for S3) or gcsfuse (for GCS) must be installed on the host. macOS also requires macFUSE. Clear error messages with install instructions are shown if tools are missing.
