#!/usr/bin/env python3
"""Package GSL dataset"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from gsl_training_pipeline.utils.dataset_packager import GSLDatasetPackager

def main():
    packager = GSLDatasetPackager()
    dataset_info = packager.package_dataset()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
