# OCR Table Extraction - Test Suite

This directory contains comprehensive test cases for the OCR Table Extraction project, focusing on the most critical functionality.

## Test Files

### Core Functionality Tests
- **`test_currency_utils.py`** - Tests money detection logic (critical for identifying budget tables)
- **`test_budget_extractor.py`** - Tests budget data extraction (the main business logic)
- **`test_excel_processor.py`** - Tests Excel/CSV file processing
- **`test_pdf_image_processor.py`** - Tests PDF/Image OCR processing

### Integration Tests
- **`test_integration.py`** - Tests complete workflows and component integration

### Test Runner
- **`run_tests.py`** - Main test runner script with detailed reporting

## Running Tests

### Run All Tests
```bash
cd tests
python run_tests.py
```

### Run Specific Test Module
```bash
cd tests
python run_tests.py currency_utils
python run_tests.py budget_extractor
```

### Run Individual Test File
```bash
python test_currency_utils.py
python test_budget_extractor.py
```

## Test Coverage

The test suite focuses on:

1. **Pandas Series Boolean Error Fixes** - Ensures the recently fixed errors don't regress
2. **Money Detection** - Core logic for identifying budget-related tables
3. **Data Cleaning** - Robust handling of messy OCR data
4. **Budget Extraction** - The main business logic for extracting financial data
5. **Edge Cases** - Empty data, malformed input, Unicode content
6. **Error Handling** - Graceful failure and recovery

## Key Test Categories

### Unit Tests
- Individual function testing
- Edge case handling
- Data type validation
- Error condition testing

### Integration Tests
- Complete workflow testing
- Component interaction
- End-to-end data flow
- Mock external dependencies

### Regression Tests
- Pandas Series boolean error prevention
- MultiIndex column handling
- Currency detection accuracy

## Test Data

Tests use synthetic data to avoid dependencies on external files:
- Sample Excel/CSV data
- Mock OCR table structures
- Various currency formats
- Multilingual content

## Dependencies

Tests are designed to run with minimal dependencies:
- `unittest` (Python standard library)
- `pandas` (already required by main project)
- Mock external services and file I/O

Most tests will run even if Azure services or other external dependencies are unavailable.
