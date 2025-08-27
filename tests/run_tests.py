"""
Test runner script to execute all tests with proper reporting.
Run this script to execute the complete test suite.
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Discover all tests in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback else 'No details'}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback else 'No details'}")
    
    if result.skipped:
        print(f"\nSKIPPED:")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run tests from a specific module."""
    try:
        module = __import__(test_module)
        suite = unittest.TestLoader().loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Could not import {test_module}: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        if not test_module.startswith('test_'):
            test_module = f'test_{test_module}'
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
