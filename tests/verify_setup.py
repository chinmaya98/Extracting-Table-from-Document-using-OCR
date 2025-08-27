"""
Simple test verification script to ensure the test suite is working.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that we can import all necessary modules."""
    try:
        from utils.currency_utils import contains_money, MONEY_PATTERN
        print("✅ Currency utils imported successfully")
        
        from budget_extractor import BudgetExtractor
        print("✅ Budget extractor imported successfully")
        
        from excel_processor import ExcelProcessor
        print("✅ Excel processor imported successfully")
        
        from pdf_image_processor import PDFImageProcessor
        print("✅ PDF image processor imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    import pandas as pd
    from utils.currency_utils import contains_money
    from budget_extractor import BudgetExtractor
    
    print("\n🧪 Testing basic functionality...")
    
    # Test currency detection
    df = pd.DataFrame({
        'Description': ['Item 1', 'Item 2'],
        'Amount': ['$100', '$200']
    })
    
    if contains_money(df):
        print("✅ Currency detection working")
    else:
        print("❌ Currency detection failed")
        return False
    
    # Test budget extraction
    extractor = BudgetExtractor()
    result = extractor._extract_budget_from_single_table(df)
    
    if not result.empty and len(result) == 2:
        print("✅ Budget extraction working")
    else:
        print("❌ Budget extraction failed")
        return False
    
    return True

if __name__ == '__main__':
    print("🔍 Verifying test suite setup...\n")
    
    if test_imports():
        print("\n📦 All imports successful!")
        
        if test_basic_functionality():
            print("\n🎉 Test suite verification complete!")
            print("✅ All core functionality working")
            print("\nYou can now run the full test suite with:")
            print("python run_tests.py")
        else:
            print("\n❌ Basic functionality tests failed")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed")
        sys.exit(1)
