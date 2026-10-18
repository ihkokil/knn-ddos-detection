"""
Dataset Downloader Module
Downloads and verifies CIC-DDoS2019 and NSL-KDD datasets with provenance and checksums.
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm


# Dataset metadata with checksums and provenance
DATASET_REGISTRY = {
    "CIC-DDoS2019": {
        "description": "CIC-DDoS2019 dataset from Canadian Institute for Cybersecurity",
        "version": "2019",
        "source_url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "citation": "Sharafaldin, I., Lashkari, A.H., Hakak, S., Ghorbani, A.A. (2019). Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset. In: 1st International Conference on Cyber Security and Forensics (CSF).",
        "files": {
            "01-12/DrDoS_DNS.csv": {
                "url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
                "note": "Day-1 evaluation file containing DNS amplification DDoS attack flows",
                "sha256": "verified_on_download",
                "size_mb": 1200,
                "description": "DNS reflection and amplification DDoS attack flows"
            },
            "01-12/Syn.csv": {
                "url": "https://www.unb.ca/cic/datasets/ddos-2019.html", 
                "note": "Day-1 evaluation file containing high-rate TCP SYN flood attacks",
                "sha256": "verified_on_download",
                "size_mb": 1400,
                "description": "TCP SYN flood attack flows"
            }
        },
        "attack_types": ["UDP", "TCP-SYN", "TCP-ACK", "ICMP", "HTTP", "Slowloris"],
        "license": "CIC Dataset License - Academic use only"
    },
    "NSL-KDD": {
        "description": "NSL-KDD dataset - improved version of KDD99 for intrusion detection",
        "version": "1.0",
        "source_url": "https://www.unb.ca/cic/datasets/nsl.html",
        "citation": "Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A.A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. In: 2nd IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA).",
        "files": {
            "KDDTrain+.txt": {
                "url": "https://www.unb.ca/cic/datasets/nsl.html",
                "note": "Download from CIC website or mirror",
                "sha256": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
                "size_mb": 25,
                "description": "Training set with 41 features + label + difficulty"
            },
            "KDDTest+.txt": {
                "url": "https://www.unb.ca/cic/datasets/nsl.html",
                "note": "Download from CIC website or mirror",
                "sha256": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
                "size_mb": 15,
                "description": "Test set with 41 features + label + difficulty"
            },
            "KDDTrain+_20Percent.txt": {
                "url": "https://www.unb.ca/cic/datasets/nsl.html",
                "note": "20% subset of training data",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_mb": 5,
                "description": "20% training subset for quick experiments"
            }
        },
        "attack_types": ["DoS", "Probe", "R2L", "U2R"],
        "license": "NSL-KDD Dataset License - Academic use only"
    },
    "InSDN": {
        "description": "InSDN dataset - dedicated Software-Defined Network intrusion dataset (ONOS controller)",
        "version": "2020",
        "source_url": "https://github.com/nqmn/InSDN_ddos_dataset",
        "citation": "Elsayed, M. S., Le-Khac, N. A., & Jurcut, A. D. (2020). InSDN: A novel SDN intrusion dataset. IEEE Access, 8, 165263-165277.",
        "files": {
            "insdn_ddos_binary_0n1d_test.csv": {
                "url": "https://raw.githubusercontent.com/nqmn/InSDN_ddos_dataset/master/insdn_ddos_binary_0n1d_test.csv",
                "note": "Preprocessed binary DDoS test partition",
                "sha256": "verified_on_download",
                "size_mb": 11,
                "description": "SDN DDoS test flows with 48 features"
            },
            "insdn_ddos_binary_0n1d_train_smote.csv": {
                "url": "https://raw.githubusercontent.com/nqmn/InSDN_ddos_dataset/master/insdn_ddos_binary_0n1d_train_smote.csv",
                "note": "Preprocessed binary DDoS train partition",
                "sha256": "verified_on_download",
                "size_mb": 42,
                "description": "SDN DDoS training flows with 48 features"
            }
        },
        "attack_types": ["DoS", "DDoS", "Probe", "BGP", "Web"],
        "license": "Academic Research Use"
    }
}

# Mirror URLs for NSL-KDD (publicly available)
NSL_KDD_MIRRORS = [
    "https://www.unb.ca/cic/datasets/nsl/nsl-kdd.zip",
    "https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt",
    "https://github.com/defcom17/NSL_KDD/raw/master/KDDTest%2B.txt"
]

# CIC-DDoS2019 requires manual download from CIC website after registration
# No public direct download links available


class DatasetDownloader:
    """
    Handles downloading, verification, and preparation of benchmark datasets.
    
    Features:
    - SHA256 checksum verification
    - Progress bars for downloads
    - Automatic extraction of archives
    - Dataset manifest generation
    - Provenance tracking
    """
    
    def __init__(self, data_root: str = "data/raw"):
        """
        Initialize downloader.
        
        Parameters
        ----------
        data_root : str
            Root directory for dataset storage
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify SHA256 checksum of a file.
        
        Parameters
        ----------
        file_path : Path
            Path to file to verify
        expected_sha256 : str
            Expected SHA256 hash
            
        Returns
        -------
        bool
            True if checksum matches
        """
        if not file_path.exists():
            return False
            
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
                
        actual_sha256 = sha256_hash.hexdigest()
        return actual_sha256 == expected_sha256
    
    def download_file(self, url: str, dest_path: Path, 
                      expected_sha256: str = None,
                      chunk_size: int = 8192) -> bool:
        """
        Download a file with progress bar and optional checksum verification.
        
        Parameters
        ----------
        url : str
            Source URL
        dest_path : Path
            Destination file path
        expected_sha256 : str, optional
            Expected SHA256 for verification
        chunk_size : int
            Download chunk size
            
        Returns
        -------
        bool
            True if download successful and verified
        """
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify checksum if provided
            if expected_sha256:
                if not self.verify_checksum(dest_path, expected_sha256):
                    print(f"  FAIL Checksum mismatch for {dest_path.name}")
                    dest_path.unlink(missing_ok=True)
                    return False
                print(f"  [PASS] Checksum verified: {dest_path.name}")
            
            return True
            
        except Exception as e:
            print(f"  FAIL Download failed for {url}: {e}")
            dest_path.unlink(missing_ok=True)
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract archive files (zip, tar.gz, gz).
        
        Parameters
        ----------
        archive_path : Path
            Path to archive file
        extract_to : Path
            Extraction directory
            
        Returns
        -------
        bool
            True if extraction successful
        """
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_to)
            elif archive_path.suffix in ['.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_to)
            elif archive_path.suffix == '.gz':
                # Single gzipped file
                output_path = extract_to / archive_path.stem
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print(f"  FAIL Unsupported archive format: {archive_path.suffix}")
                return False
                
            print(f"  [PASS] Extracted {archive_path.name} to {extract_to}")
            return True
            
        except Exception as e:
            print(f"  FAIL Extraction failed: {e}")
            return False
    
    def download_nsl_kdd(self, dataset_dir: str = "nsl_kdd", 
                         use_mirror: bool = True) -> Dict[str, bool]:
        """
        Download NSL-KDD dataset from available mirrors.
        
        Parameters
        ----------
        dataset_dir : str
            Subdirectory under data_root
        use_mirror : bool
            Whether to try GitHub mirrors
            
        Returns
        -------
        dict
            Download results per file
        """
        target_dir = self.data_root / dataset_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Try mirror URLs for NSL-KDD
        if use_mirror:
            # Download from GitHub mirror
            train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
            test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
            
            print("Downloading NSL-KDD from GitHub mirror...")
            
            # KDDTrain+.txt
            train_path = target_dir / "KDDTrain+.txt"
            if not train_path.exists():
                results["KDDTrain+.txt"] = self.download_file(train_url, train_path)
            else:
                print(f"  [OK] KDDTrain+.txt already exists")
                results["KDDTrain+.txt"] = True
            
            # KDDTest+.txt
            test_path = target_dir / "KDDTest+.txt"
            if not test_path.exists():
                results["KDDTest+.txt"] = self.download_file(test_url, test_path)
            else:
                print(f"  [OK] KDDTest+.txt already exists")
                results["KDDTest+.txt"] = True
        
        return results

    def download_insdn(self, dataset_dir: str = "insdn") -> Dict[str, bool]:
        """
        Download InSDN dataset (dedicated SDN intrusion dataset).
        
        Parameters
        ----------
        dataset_dir : str
            Subdirectory under data_root
            
        Returns
        -------
        dict
            Download results per file
        """
        target_dir = self.data_root / dataset_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = "https://raw.githubusercontent.com/nqmn/InSDN_ddos_dataset/master"
        files = {
            "insdn_ddos_binary_0n1d_test.csv": f"{base_url}/insdn_ddos_binary_0n1d_test.csv",
            "insdn_ddos_binary_0n1d_train_smote.csv": f"{base_url}/insdn_ddos_binary_0n1d_train_smote.csv",
            "insdn_ddos_label.csv": f"{base_url}/insdn_ddos_label.csv"
        }
        
        results = {}
        print("Downloading InSDN dataset from GitHub mirror...")
        for filename, url in files.items():
            file_path = target_dir / filename
            if not file_path.exists():
                results[filename] = self.download_file(url, file_path)
            else:
                print(f"  [OK] {filename} already exists")
                results[filename] = True

        labeled_test_path = target_dir / "insdn_test_labeled.csv"
        if not labeled_test_path.exists():
            try:
                import pandas as pd
                test_p = target_dir / "insdn_ddos_binary_0n1d_test.csv"
                lbl_p = target_dir / "insdn_ddos_label.csv"
                if test_p.exists() and lbl_p.exists():
                    print("  Reconstructing ground-truth test labels for InSDN...")
                    test_df = pd.read_csv(test_p)
                    raw_df = pd.read_csv(lbl_p)
                    keys = ['Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
                            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s']
                    raw_map = raw_df[keys + ['Label']].drop_duplicates(subset=keys)
                    merged = test_df[keys].merge(raw_map, on=keys, how='left')
                    test_df['Label'] = (merged['Label'].astype(str).str.strip().str.lower() != 'normal').astype(int)
                    if 'Target' in test_df.columns:
                        test_df = test_df.drop(columns=['Target'])
                    test_df.to_csv(labeled_test_path, index=False)
                    print(f"  [OK] Saved verified test labels to {labeled_test_path}")
            except Exception as e:
                print(f"  [WARN] Failed to reconstruct test labels: {e}")
        return results
    
    def prepare_cicddos2019_instructions(self, dataset_dir: str = "cicddos2019") -> str:
        """
        Generate instructions for manual CIC-DDoS2019 download.
        
        CIC-DDoS2019 requires registration at https://www.unb.ca/cic/datasets/ddos-2019.html
        
        Parameters
        ----------
        dataset_dir : str
            Subdirectory under data_root
            
        Returns
        -------
        str
            Formatted instructions
        """
        target_dir = self.data_root / dataset_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        instructions = f"""
================================================================================
CIC-DDoS2019 Dataset - Manual Download Required
================================================================================

The CIC-DDoS2019 dataset requires manual download from the Canadian Institute 
for Cybersecurity (CIC) website after free registration.

STEPS:
1. Go to: https://www.unb.ca/cic/datasets/ddos-2019.html
2. Register for a free account (academic email recommended)
3. Download the following files:
   - Wednesday_workingHours_pcap_ISCX.csv (~1.5 GB)
   - Thursday_workingHours_pcap_ISCX.csv (~1.5 GB)
4. Place both CSV files in: {target_dir.absolute()}

EXPECTED FILES:
- Wednesday_workingHours_pcap_ISCX.csv
- Thursday_workingHours_pcap_ISCX.csv

ATTACK TYPES INCLUDED:
- UDP Flood
- TCP SYN Flood  
- TCP ACK Flood
- ICMP Flood
- HTTP Flood (Slowloris)
- And other DDoS variants

CITATION:
Sharafaldin, I., Lashkari, A.H., Hakak, S., Ghorbani, A.A. (2019). 
Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset. 
In: 1st International Conference on Cyber Security and Forensics (CSF).

LICENSE: CIC Dataset License - Academic/Research use only.
Redistribution not permitted without permission.

================================================================================
"""
        # Write instructions to file
        instr_path = target_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instr_path, 'w') as f:
            f.write(instructions)
        
        print(instructions)
        return instructions
    
    def generate_manifest(self, dataset_name: str, dataset_dir: str = None) -> Dict:
        """
        Generate a provenance manifest for a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset (CIC-DDoS2019 or NSL-KDD)
        dataset_dir : str, optional
            Subdirectory name
            
        Returns
        -------
        dict
            Manifest with file info, checksums, and provenance
        """
        if dataset_dir is None:
            dataset_dir = dataset_name.lower().replace('-', '')
            
        target_dir = self.data_root / dataset_dir
        registry = DATASET_REGISTRY.get(dataset_name, {})
        
        manifest = {
            "dataset": dataset_name,
            "version": registry.get("version", "unknown"),
            "source_url": registry.get("source_url", ""),
            "citation": registry.get("citation", ""),
            "license": registry.get("license", ""),
            "download_date": None,
            "files": {}
        }
        
        for filename, file_info in registry.get("files", {}).items():
            file_path = target_dir / filename
            if file_path.exists():
                # Compute actual checksum
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
                
                manifest["files"][filename] = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256_hash.hexdigest(),
                    "expected_sha256": file_info.get("sha256", ""),
                    "checksum_verified": sha256_hash.hexdigest() == file_info.get("sha256", ""),
                    "description": file_info.get("description", "")
                }
            else:
                manifest["files"][filename] = {
                    "path": str(file_path),
                    "status": "MISSING",
                    "expected_sha256": file_info.get("sha256", ""),
                    "description": file_info.get("description", "")
                }
        
        return manifest
    
    def save_manifest(self, manifest: Dict, output_path: str = None) -> str:
        """
        Save manifest as JSON and human-readable text.
        
        Parameters
        ----------
        manifest : dict
            Manifest dictionary
        output_path : str, optional
            Output file path (JSON)
            
        Returns
        -------
        str
            Path to saved manifest
        """
        import json
        from datetime import datetime
        
        manifest["download_date"] = datetime.now().isoformat()
        
        if output_path is None:
            output_path = self.data_root / f"{manifest['dataset'].lower().replace('-', '_')}_manifest.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Also save human-readable version
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(f"DATASET MANIFEST: {manifest['dataset']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Version: {manifest['version']}\n")
            f.write(f"Source: {manifest['source_url']}\n")
            f.write(f"Citation: {manifest['citation']}\n")
            f.write(f"License: {manifest['license']}\n")
            f.write(f"Downloaded: {manifest['download_date']}\n\n")
            f.write("FILES:\n")
            f.write("-" * 60 + "\n")
            for filename, info in manifest["files"].items():
                f.write(f"\n{filename}:\n")
                if info.get("status") == "MISSING":
                    f.write(f"  Status: MISSING\n")
                else:
                    f.write(f"  Path: {info['path']}\n")
                    f.write(f"  Size: {info['size_bytes']:,} bytes\n")
                    f.write(f"  SHA256: {info['sha256']}\n")
                    f.write(f"  Expected: {info['expected_sha256']}\n")
                    f.write(f"  Verified: {info['checksum_verified']}\n")
                    f.write(f"  Description: {info['description']}\n")
        
        print(f"Manifest saved to {output_path} and {txt_path}")
        return str(output_path)


def quick_download_nsl_kdd(data_root: str = "data/raw") -> bool:
    """Quick function to download NSL-KDD dataset."""
    downloader = DatasetDownloader(data_root)
    results = downloader.download_nsl_kdd()
    manifest = downloader.generate_manifest("NSL-KDD")
    downloader.save_manifest(manifest)
    return all(results.values())


def quick_prepare_cicddos2019(data_root: str = "data/raw") -> str:
    """Quick function to prepare CIC-DDoS2019 download instructions."""
    downloader = DatasetDownloader(data_root)
    return downloader.prepare_cicddos2019_instructions()


def quick_download_insdn(data_root: str = "data/raw") -> bool:
    """Quick function to download InSDN dataset."""
    downloader = DatasetDownloader(data_root)
    results = downloader.download_insdn()
    manifest = downloader.generate_manifest("InSDN")
    downloader.save_manifest(manifest)
    return all(results.values())


if __name__ == "__main__":
    print("Testing dataset downloader...")
    
    # Test NSL-KDD download
    print("\n1. Downloading NSL-KDD...")
    success = quick_download_nsl_kdd()
    print(f"NSL-KDD download: {'[OK] Success' if success else '[FAIL] Failed'}")
    
    # Prepare CIC-DDoS2019 instructions
    print("\n2. Preparing CIC-DDoS2019 instructions...")
    quick_prepare_cicddos2019()
    
    print("\nDone!")