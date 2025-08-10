import langextract as lx

# Test basic functionality
print("Available functions:")
print([attr for attr in dir(lx) if not attr.startswith('_')])

# Test if we can access the extract function
try:
    extract_func = lx.extract
    print("Successfully accessed extract function")
except AttributeError as e:
    print("Failed to access extract function:", e)

# Test if we can access the data module
try:
    from langextract import data
    print("Successfully imported data module")
    # Test creating an ExampleData object
    example = data.ExampleData(text="Test text", extractions=[])
    print("Successfully created ExampleData object")
except Exception as e:
    print("Failed to work with data module:", e)