# Shared Conda Environments Guide

This guide explains how to create and manage shared Conda environments in multi-user systems, JupyterHub environments using the Amazon EFS network file system.

## Conda Environment Locations

To check where Conda looks for environments, use:

```bash
conda config --show envs_dirs
```

Typical locations include:
- Personal environments: `/home/jovyan/.conda/envs/`
- Shared environments: `/efs/shared/.conda/envs/`

## Creating a Shared Environment

### 1. Specify the Prefix in YAML

In your environment YAML file (e.g., `environment.yml`), add a prefix pointing to the shared location:

```yaml
name: lmm_poc_env
channels:
  - defaults
dependencies:
  - python=3.12
  - pytorch=2.1.0
  # ... other dependencies
prefix: /efs/shared/.conda/envs/lmm_poc_env
```

### 2. Create Environment Using Prefix

```bash
conda env create -f environment.yml --prefix /efs/shared/.conda/envs/lmm_poc_env
```

### 3. Update Existing Environment

When you need to update dependencies:

```bash
conda env update -f environment.yml --prefix /efs/shared/.conda/envs/lmm_poc_env --prune
```

The `--prune` flag removes dependencies that are no longer specified in the YAML file.

## Setting Permissions for Team Access

**IMPORTANT**: Always set group ownership **before** applying permissions. Running `chmod 2770` while the group is `root` will lock out all team members.

### Step 1: Verify Current Ownership

```bash
ls -ld /efs/shared/.conda/envs/lmm_poc_env
```

### Step 2: Check Your Team Group

```bash
groups
```

### Step 3: Set Group Ownership First

```bash
chgrp -R users /efs/shared/.conda/envs/lmm_poc_env
```

### Step 4: Then Apply Permissions

```bash
chmod -R 2770 /efs/shared/.conda/envs/lmm_poc_env
```

| Permission Digit | Meaning |
|-----------------|---------|
| `2` | setgid — new files inherit the directory's group |
| `7` | owner: read + write + execute |
| `7` | group: read + write + execute |
| `0` | others: no access |

The setgid bit (`2`) ensures **future** files inherit the group. The `chgrp -R` fixes group on **existing** contents.

## Using the Shared Environment

### 1. Method 1: Activating by Path

Users can activate the environment using its full path:

```bash
conda activate /efs/shared/.conda/envs/lmm_poc_env
```

### 2. Method 2: Adding to Known Environments

Users can add the shared directory to their environment search path:

```bash
# Add shared environment location to config
conda config --append envs_dirs /efs/shared/.conda/envs

# Now users can activate by name
conda activate lmm_poc_env
```

## Package Management and Updates

### Administrator Responsibilities

1. **Schedule Regular Updates**:
   ```bash
   # First test the updated environment in a test location
   conda env update -f environment.yml --prefix /efs/shared/.conda/envs/lmm_poc_env_test --prune

   # Once tested, update the production environment
   conda env update -f environment.yml --prefix /efs/shared/.conda/envs/lmm_poc_env --prune
   ```

2. **Track Dependencies**:
   ```bash
   # Export current environment to a requirements file for reference
   conda list --explicit > lmm_poc_env_snapshot_$(date +%Y%m%d).txt
   ```

3. **Custom Channel Configuration**:
   ```bash
   # Add internal channels when needed
   conda config --add channels https://your-internal-channel.example.com
   ```

### Best Practices

1. **Permissions Management**:
   ```bash
   # Always set group BEFORE permissions
   chgrp -R users /efs/shared/.conda/envs/lmm_poc_env
   chmod -R 2770 /efs/shared/.conda/envs/lmm_poc_env
   ```

2. **Communication with Users**:
   - Establish a notification system for environment changes
   - Provide clear documentation on how to use the environment
   - Set up a regular maintenance schedule

3. **Version Control**:
   - Keep all environment files under version control
   - Document changes with each update
   - Consider maintaining multiple environment versions for compatibility

4. **Validation**:
   ```bash
   # Add a validation script to test that all core functionality works
   python -m internvl.utils.verify_env
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Permission Errors**:
   ```bash
   # Fix permission issues — always set group first
   chgrp -R users /efs/shared/.conda/envs/lmm_poc_env
   chmod -R 2770 /efs/shared/.conda/envs/lmm_poc_env
   ```

2. **Package Conflicts**:
   ```bash
   # Use the --no-deps flag when installing individual packages
   conda install --no-deps package_name
   ```

3. **Identifying Issues**:
   ```bash
   # Check environment integrity
   conda list --revisions
   ```

4. **libtinfow.so.6 Missing (ncurses wide-character issue)**:

   After activating a conda environment, system commands like `reset` or `clear` may fail with:
   ```
   reset: error while loading shared libraries: libtinfow.so.6: cannot open shared object file
   ```

   **Cause**: Conda's `ncurses` package may not include the wide-character variant (`libtinfow`). Conda prepends its `lib/` directory to `LD_LIBRARY_PATH`, shadowing the system libraries.

   **Fix**: Use the system binaries and add aliases to `~/.bashrc`:
   ```bash
   echo 'alias reset="/usr/bin/reset"' >> ~/.bashrc
   echo 'alias clear="/usr/bin/clear"' >> ~/.bashrc
   source ~/.bashrc
   ```

   **Alternative**: If `conda install ncurses -c conda-forge` is available, it may provide the wide-character build. Otherwise the aliases above are the reliable workaround.

## Example for LMM_POC Project

Create a shared environment specifically for the LMM_POC project:

```bash
# Configure conda to use shared environment directory
conda config --append envs_dirs /efs/shared/.conda/envs

# Create the shared environment
conda env create -f environment.yml --prefix /efs/shared/.conda/envs/lmm_poc_env

# Make sure group ownership is correct FIRST
chgrp -R users /efs/shared/.conda/envs/lmm_poc_env

# Then set permissions
chmod -R 2770 /efs/shared/.conda/envs/lmm_poc_env

# Create a validation script that users can run
echo "python -m internvl.utils.verify_env" > /efs/shared/.conda/envs/validate_lmm_poc.sh
chmod +x /efs/shared/.conda/envs/validate_lmm_poc.sh
```

## Create an ipynb Notebook Kernel for the shared environment

```bash
python -m ipykernel install --user --name lmm_poc_env --display-name "POC Environment"
```

This approach ensures a consistent, well-maintained environment that all users can access while minimizing duplication and conflicts.
