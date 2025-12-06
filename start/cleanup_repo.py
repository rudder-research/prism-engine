#!/usr/bin/env python3
"""
PRISM Repo Cleanup Script
=========================

Safely cleans up duplicate/orphan files after checking import dependencies.

Usage:
    python cleanup_repo.py --check    # Check only, don't delete
    python cleanup_repo.py --clean    # Actually clean up
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent
BACKUP_DIR = PROJECT_ROOT / "_cleanup_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")


def find_imports(file_path: Path) -> list:
    """Find all imports in a Python file."""
    imports = []
    try:
        content = file_path.read_text()
        # Match: from X import Y, import X
        patterns = [
            r'from\s+([\w.]+)\s+import',
            r'^import\s+([\w.]+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
    except:
        pass
    return imports


def check_module_usage(module_name: str) -> dict:
    """Check how many files import this module."""
    results = {'importers': [], 'count': 0}
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip backup dirs, venv, cache
        if any(skip in str(py_file) for skip in ['_legacy', 'venv', '__pycache__', '_cleanup']):
            continue
        
        try:
            content = py_file.read_text()
            if module_name in content:
                results['importers'].append(str(py_file.relative_to(PROJECT_ROOT)))
                results['count'] += 1
        except:
            pass
    
    return results


def analyze_duplicates():
    """Analyze duplicate modules."""
    print("\n" + "="*60)
    print("📋 DUPLICATE MODULE ANALYSIS")
    print("="*60)
    
    duplicates = [
        {
            'files': ['data/sql/db_connector.py', 'utils/db_connector.py'],
            'imports': ['data.sql.db_connector', 'utils.db_connector'],
        },
        {
            'files': ['data/sql/db.py', 'data/sql/prism_db.py'],
            'imports': ['data.sql.db', 'data.sql.prism_db'],
        },
        {
            'files': ['fetch/fetcher_climate.py', 'fetch/fetch_climate.py'],
            'imports': ['fetch.fetcher_climate', 'fetch.fetch_climate'],
        },
        {
            'files': ['engine/prism_ml_engine.py', 'ml/prism_ml_engine.py'],
            'imports': ['engine.prism_ml_engine', 'ml.prism_ml_engine'],
        },
        {
            'files': ['yaml_loader.py', 'data/registry/yaml_loader.py'],
            'imports': ['yaml_loader', 'data.registry.yaml_loader'],
        },
    ]
    
    for dup in duplicates:
        print(f"\n🔍 Checking: {dup['files']}")
        for i, (file, imp) in enumerate(zip(dup['files'], dup['imports'])):
            usage = check_module_usage(imp)
            exists = (PROJECT_ROOT / file).exists()
            status = "✅ EXISTS" if exists else "❌ MISSING"
            print(f"   {status} {file}")
            print(f"      Import '{imp}' used by {usage['count']} files")
            if usage['count'] > 0 and usage['count'] <= 5:
                for f in usage['importers'][:5]:
                    print(f"         - {f}")


def analyze_orphans():
    """Analyze orphan files that should be deleted."""
    print("\n" + "="*60)
    print("🗑️  ORPHAN FILE ANALYSIS")
    print("="*60)
    
    orphans = [
        './__init__.py',
        './fetcher_PATCHED.py',
        './fetcher_yahoo_PATCHED.py',
        './start/run_overnight_analysis-old.py',
        './fetch/fetch_all_hybrid.py',
        './fetch/fetch_all_stooq.py',
    ]
    
    for orphan in orphans:
        path = PROJECT_ROOT / orphan.lstrip('./')
        exists = path.exists()
        
        if exists:
            # Check if anything imports it
            module_name = orphan.replace('./', '').replace('/', '.').replace('.py', '')
            usage = check_module_usage(module_name)
            
            if usage['count'] == 0:
                print(f"   ✅ SAFE TO DELETE: {orphan} (no imports)")
            else:
                print(f"   ⚠️  KEEP: {orphan} (used by {usage['count']} files)")
                for f in usage['importers'][:3]:
                    print(f"         - {f}")
        else:
            print(f"   ⏭️  SKIP: {orphan} (doesn't exist)")


def analyze_movable():
    """Analyze files that should be moved."""
    print("\n" + "="*60)
    print("📦 FILES TO MOVE")
    print("="*60)
    
    moves = [
        {'from': './loader.py', 'to': './utils/loader.py'},
    ]
    
    for move in moves:
        from_path = PROJECT_ROOT / move['from'].lstrip('./')
        to_path = PROJECT_ROOT / move['to'].lstrip('./')
        
        if from_path.exists():
            # Check imports
            module_name = move['from'].replace('./', '').replace('.py', '')
            usage = check_module_usage(module_name)
            
            print(f"   {move['from']} → {move['to']}")
            print(f"      Currently imported by {usage['count']} files")
            if usage['count'] > 0:
                print(f"      ⚠️  Will need to update imports!")
                for f in usage['importers'][:5]:
                    print(f"         - {f}")
        else:
            print(f"   ⏭️  SKIP: {move['from']} (doesn't exist)")


def safe_delete(file_path: Path, backup: bool = True):
    """Safely delete a file (with backup)."""
    if not file_path.exists():
        return False
    
    if backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / file_path.name
        shutil.copy2(file_path, backup_path)
        print(f"   Backed up: {backup_path}")
    
    file_path.unlink()
    print(f"   Deleted: {file_path}")
    return True


def safe_move(from_path: Path, to_path: Path, backup: bool = True):
    """Safely move a file (with backup)."""
    if not from_path.exists():
        return False
    
    if backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / from_path.name
        shutil.copy2(from_path, backup_path)
    
    to_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(from_path, to_path)
    print(f"   Moved: {from_path} → {to_path}")
    return True


def do_cleanup():
    """Perform the actual cleanup."""
    print("\n" + "="*60)
    print("🧹 PERFORMING CLEANUP")
    print("="*60)
    print(f"   Backup directory: {BACKUP_DIR}")
    
    # 1. Delete orphans (only if no imports)
    safe_orphans = [
        './__init__.py',
        './fetcher_PATCHED.py', 
        './fetcher_yahoo_PATCHED.py',
        './start/run_overnight_analysis-old.py',
    ]
    
    print("\n📍 Deleting orphan files...")
    for orphan in safe_orphans:
        path = PROJECT_ROOT / orphan.lstrip('./')
        if path.exists():
            # Double-check no imports
            module_name = orphan.replace('./', '').replace('/', '.').replace('.py', '')
            usage = check_module_usage(module_name)
            if usage['count'] == 0:
                safe_delete(path)
            else:
                print(f"   ⏭️  Skipping {orphan} (still has {usage['count']} imports)")
    
    # 2. Handle yaml_loader duplicate
    print("\n📍 Handling yaml_loader duplicate...")
    root_yaml = PROJECT_ROOT / 'yaml_loader.py'
    if root_yaml.exists():
        # Check if root version is imported
        usage = check_module_usage('yaml_loader')
        if usage['count'] == 0:
            safe_delete(root_yaml)
        else:
            print(f"   ⚠️  yaml_loader.py has {usage['count']} imports - update them first")
            for f in usage['importers']:
                print(f"      sed -i '' 's/from yaml_loader/from data.registry.yaml_loader/g' {f}")
    
    # 3. Update .gitignore
    print("\n📍 Updating .gitignore...")
    gitignore_additions = """
# ===== PRISM CLEANUP =====
# Virtual environments - NEVER track
_legacy_backup/
_cleanup_backup/
*venv*/
venv/
.venv/
env/

# Output files - generated, not source
output/
*.csv
*.png
*.db
*.log
overnight.log

# Data files - too large
data/panels/*.csv
prism_data/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Mac files
.DS_Store
**/.DS_Store

# IDE
.idea/
.vscode/
*.swp
"""
    
    gitignore_path = PROJECT_ROOT / '.gitignore'
    
    # Check if already has our additions
    existing = gitignore_path.read_text() if gitignore_path.exists() else ""
    if "PRISM CLEANUP" not in existing:
        with open(gitignore_path, 'a') as f:
            f.write(gitignore_additions)
        print("   ✅ Updated .gitignore")
    else:
        print("   ⏭️  .gitignore already updated")
    
    print("\n" + "="*60)
    print("✅ CLEANUP COMPLETE")
    print("="*60)
    print(f"   Backup location: {BACKUP_DIR}")
    print("\n   Next steps:")
    print("   1. Run: git status")
    print("   2. Run: git add -A && git commit -m 'Clean up repo structure'")
    print("   3. Test: python start/fetcher.py --test")


def main():
    parser = argparse.ArgumentParser(description="PRISM Repo Cleanup")
    parser.add_argument('--check', action='store_true', help='Check only, no changes')
    parser.add_argument('--clean', action='store_true', help='Actually perform cleanup')
    args = parser.parse_args()
    
    print("="*60)
    print("🔧 PRISM REPO CLEANUP TOOL")
    print("="*60)
    
    # Always run analysis
    analyze_duplicates()
    analyze_orphans()
    analyze_movable()
    
    if args.clean:
        do_cleanup()
    else:
        print("\n" + "="*60)
        print("ℹ️  DRY RUN - No changes made")
        print("="*60)
        print("   Run with --clean to perform cleanup")
        print("   Example: python cleanup_repo.py --clean")


if __name__ == "__main__":
    main()
