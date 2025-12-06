#!/bin/bash
# PRISM Safe Cleanup Script
# Run from: ~/prism-engine

set -e  # Exit on error

echo "=============================================="
echo "🧹 PRISM SAFE CLEANUP"
echo "=============================================="

cd ~/prism-engine

# Create backup folder
BACKUP="_cleanup_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
echo "📁 Backup folder: $BACKUP"

# 1. Delete confirmed orphans (backup first)
echo ""
echo "📍 Removing orphan files..."

ORPHANS=(
    "start/run_overnight_analysis-old.py"
    "fetch/fetch_all_hybrid.py"
    "fetch/fetch_all_stooq.py"
    "fetcher_PATCHED.py"
    "fetcher_yahoo_PATCHED.py"
    "yamlstoqcodes.zip"
    "prism_unified_patch.diff"
    "requirements_PATCHED.txt"
    "PRISM_PATCH_PLAN.md"
)

for file in "${ORPHANS[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP/"
        rm "$file"
        echo "   ✅ Deleted: $file"
    else
        echo "   ⏭️  Skip (not found): $file"
    fi
done

# 2. Remove duplicate cleanup script in start/
if [ -f "start/cleanup_repo.py" ]; then
    rm "start/cleanup_repo.py"
    echo "   ✅ Deleted: start/cleanup_repo.py"
fi

# 3. Delete _legacy_backup if it exists
if [ -d "_legacy_backup" ]; then
    echo ""
    echo "📍 Removing _legacy_backup (old venv)..."
    rm -rf "_legacy_backup"
    echo "   ✅ Deleted: _legacy_backup/"
fi

# 4. Update .gitignore
echo ""
echo "📍 Updating .gitignore..."

if ! grep -q "PRISM CLEANUP" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'GITIGNORE'

# ===== PRISM CLEANUP (added by cleanup script) =====
# Virtual environments
_legacy_backup/
_cleanup_backup*/
*venv*/
venv/
.venv/
env/

# Output files - generated
output/
*.csv
*.png  
*.db
*.log
overnight.log

# Large data
prism_data/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Mac
.DS_Store
**/.DS_Store

# IDE
.idea/
.vscode/
*.swp

# Temp/patch files
*.diff
*.patch
*_PATCHED.py
GITIGNORE
    echo "   ✅ Updated .gitignore"
else
    echo "   ⏭️  .gitignore already updated"
fi

# 5. Show what's left to decide on manually
echo ""
echo "=============================================="
echo "📋 REMAINING DUPLICATES (decide later)"
echo "=============================================="
echo ""
echo "These have real imports - don't delete without updating code:"
echo ""
echo "   yaml_loader.py (root) vs data/registry/yaml_loader.py"
echo "   → Both used. Keep for now."
echo ""
echo "   utils/db_connector.py vs data/sql/db_connector.py"  
echo "   → Both used. Keep for now."
echo ""
echo "   data/sql/db.py vs data/sql/prism_db.py"
echo "   → Both heavily used (10+ and 8+ imports). Keep both."
echo ""
echo "   engine/prism_ml_engine.py vs ml/prism_ml_engine.py"
echo "   → Keep both for now."
echo ""

# 6. Summary
echo "=============================================="
echo "✅ CLEANUP COMPLETE"
echo "=============================================="
echo ""
echo "Backup location: $BACKUP"
echo ""
echo "Next steps:"
echo "   1. git status"
echo "   2. git add -A"
echo "   3. git commit -m 'Clean up orphan files and update gitignore'"
echo "   4. git push"
echo ""
echo "To restore if needed:"
echo "   cp $BACKUP/* ."
