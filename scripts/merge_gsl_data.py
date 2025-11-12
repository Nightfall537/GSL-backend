"""
Merge GSL Data Files

Combines all GSL sign data files into a single unified format
that the text-to-sign model can use.
"""

import json
from pathlib import Path
from typing import List, Dict


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_sign_format(sign: Dict, category: str) -> Dict:
    """Convert sign from source format to model format."""
    return {
        "word": sign.get("term", "").lower().replace(" ", "_"),
        "term": sign.get("term", ""),
        "video_url": f"/videos/signs/{category}/{sign.get('term', '').lower().replace(' ', '_')}.mp4",
        "thumbnail_url": f"/videos/thumbs/{category}/{sign.get('term', '').lower().replace(' ', '_')}.jpg",
        "description": sign.get("source_description", sign.get("movement_description", "")),
        "category": category,
        "difficulty": 1,  # Default difficulty
        "page": sign.get("page"),
        "hands_used": sign.get("hands_used"),
        "handshape": sign.get("dominant_handshape"),
        "movement": sign.get("movement_type"),
        "location": sign.get("start_location"),
        "full_details": sign  # Keep original data
    }


def merge_all_gsl_data(models_dir: str = "app/models") -> Dict:
    """Merge all GSL data files into one."""
    models_path = Path(models_dir)
    
    # Data files to merge
    data_files = {
        "colors": "colors_signs_data.json",
        "animals": "animals_signs_data.json",
        "family": "family_signs_data.json",
        "food": "food_signs_data.json",
        "grammar": "grammar_signs_data.json",
        "home_clothing": "home_clothing_signs_data.json"
    }
    
    merged_signs = []
    categories_count = {}
    
    for category, filename in data_files.items():
        file_path = models_path / filename
        
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
        
        try:
            data = load_json_file(file_path)
            signs = data.get("signs", [])
            
            print(f"✓ Loading {filename}: {len(signs)} signs")
            
            for sign in signs:
                converted = convert_sign_format(sign, category)
                merged_signs.append(converted)
            
            categories_count[category] = len(signs)
            
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
    
    # Create merged data structure
    merged_data = {
        "version": "1.0",
        "description": "Unified Ghanaian Sign Language Dictionary",
        "total_signs": len(merged_signs),
        "categories": categories_count,
        "signs": merged_signs
    }
    
    return merged_data


def save_merged_data(data: Dict, output_path: str = "app/models/gsl_unified_dictionary.json"):
    """Save merged data to file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved unified dictionary to: {output_path}")


def create_word_index(data: Dict, output_path: str = "app/models/gsl_word_index.json"):
    """Create a quick lookup index by word."""
    index = {}
    
    for sign in data["signs"]:
        word = sign["word"]
        index[word] = {
            "term": sign["term"],
            "category": sign["category"],
            "video_url": sign["video_url"],
            "description": sign["description"]
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved word index to: {output_path}")


def print_statistics(data: Dict):
    """Print statistics about the merged data."""
    print("\n" + "=" * 60)
    print("GSL UNIFIED DICTIONARY STATISTICS")
    print("=" * 60)
    print(f"\nTotal Signs: {data['total_signs']}")
    print(f"\nSigns by Category:")
    for category, count in data['categories'].items():
        print(f"  {category.capitalize()}: {count} signs")
    
    # Sample signs
    print(f"\nSample Signs:")
    for sign in data['signs'][:5]:
        print(f"  - {sign['term']} ({sign['category']})")


def main():
    """Main function."""
    print("=" * 60)
    print("MERGING GSL DATA FILES")
    print("=" * 60)
    print()
    
    # Merge all data
    merged_data = merge_all_gsl_data()
    
    # Save merged data
    save_merged_data(merged_data)
    
    # Create word index
    create_word_index(merged_data)
    
    # Print statistics
    print_statistics(merged_data)
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print("\nYou can now use the unified dictionary:")
    print("  model = TextToSignModel('app/models/gsl_unified_dictionary.json')")


if __name__ == "__main__":
    main()