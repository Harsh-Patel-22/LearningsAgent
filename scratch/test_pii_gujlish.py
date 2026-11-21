import asyncio
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pii_masker import PIIMasker

async def run_tests():
    print("=" * 60)
    print("RUNNING CONTEXT-PRESERVING DE-IDENTIFICATION TESTS")
    print("=" * 60)
    
    masker = PIIMasker()
    
    # Test 1: Basic text masking with surrogates and shifted ages
    test_text_1 = "My name is Harsh Patel. I met Ramesh yesterday. He is 28 years old and his gender is male. You can email him at ramesh@gmail.com or call at +91 98765 43210."
    print("\n--- Test 1: Context-Preserving Masking ---")
    print(f"Original: {test_text_1}")
    masked, mapping = masker.mask_text(test_text_1)
    print(f"Masked  : {masked}")
    print(f"Mapping : {mapping}")
    
    # Asserts to check context preservation
    assert "[Vikram]" in masked, "First name surrogate [Vikram] not found!"
    assert "[Sharma]" in masked, "Last name surrogate [Sharma] not found!"
    assert "[Sanjay]" in masked, "Second name surrogate [Sanjay] not found!"
    assert "[31_years_old]" in masked, "Age shift (+3 years) to 31 not found!"
    assert "[male]" in masked, "Gender [male] not found!"
    assert "[user1@mockmail.com]" in masked, "Email placeholder not found!"
    assert "[+91-90000-10001]" in masked, "Phone placeholder not found!"

    unmasked = masker.unmask_text(masked, mapping)
    print(f"Unmasked: {unmasked}")
    assert unmasked == test_text_1, "Unmasked text does not match original!"
    print("Test 1 Passed!")

    # Test 2: GujLish & Name Masking
    test_text_2 = "Kem che Amit? Maru kaam chhe ne? I spoke to Amit. He is 21 yo. We met at his office."
    print("\n--- Test 2: GujLish & Name Masking ---")
    print(f"Original: {test_text_2}")
    masked_2, mapping_2 = masker.mask_text(test_text_2)
    print(f"Masked  : {masked_2}")
    print(f"Mapping : {mapping_2}")
    
    assert "[Amit]" not in masked_2, "Original name Amit should be masked!"
    assert "[Vikram]" in masked_2, "Amit should be mapped to the first surrogate in this clean session!"
    assert "[24_yo]" in masked_2, "Age shift (+3 years) to 24 not found!"
    
    unmasked_2 = masker.unmask_text(masked_2, mapping_2)
    print(f"Unmasked: {unmasked_2}")
    assert unmasked_2 == test_text_2, "Unmasked text does not match original!"
    print("Test 2 Passed!")

    # Test 3: JSON parsing compatibility
    print("\n--- Test 3: JSON Value Masking/Unmasking ---")
    original_json = {
        "title": "Discussion with Harsh",
        "content": "Harsh Patel is 28 years old. He mentioned that Amit is also helping him.",
        "people_aliases": [{"alias": "Harsh", "role": "founder"}, {"alias": "Amit", "role": "developer"}],
        "salience": 0.85
    }
    json_str = json.dumps(original_json)
    print(f"Original JSON: {json_str}")
    
    masked_json, mapping_json = masker.mask_text(json_str)
    print(f"Masked JSON  : {masked_json}")
    print(f"Mapping      : {mapping_json}")
    
    unmasked_json = masker.unmask_text(masked_json, mapping_json)
    print(f"Unmasked JSON: {unmasked_json}")
    
    unmasked_obj = json.loads(unmasked_json)
    assert unmasked_obj == original_json, "Unmasked JSON object does not match original!"
    print("Test 3 Passed!")

    # Test 4: Streaming buffer unmasking simulator
    print("\n--- Test 4: Streaming Buffer Simulator ---")
    chunks = [
        "Response: ", "I suggest ", "talking to ", "[Vik", "ram]", " who is ", "[31_ye", "ars_old] today."
    ]
    simulated_mapping = {"[Vikram]": "Harsh Patel", "[31_years_old]": "28 years old"}
    
    full_text = ""
    yielded_len = 0
    yielded_chunks = []
    
    for chunk in chunks:
        full_text += chunk
        unmasked = masker.unmask_text(full_text, simulated_mapping)
        
        # logic from stream_respond
        last_bracket = unmasked.rfind("[")
        if last_bracket != -1 and "]" not in unmasked[last_bracket:]:
            to_yield = unmasked[yielded_len:last_bracket]
            yielded_len = last_bracket
        else:
            to_yield = unmasked[yielded_len:]
            yielded_len = len(unmasked)
            
        if to_yield:
            yielded_chunks.append(to_yield)
            print(f"Chunk yielded: {repr(to_yield)}")
            
    # Yield remaining
    unmasked = masker.unmask_text(full_text, simulated_mapping)
    if yielded_len < len(unmasked):
        yielded_chunks.append(unmasked[yielded_len:])
        print(f"Final Chunk yielded: {repr(unmasked[yielded_len:])}")
        
    final_output = "".join(yielded_chunks)
    expected_output = "Response: I suggest talking to Harsh Patel who is 28 years old today."
    print(f"Combined Output: {final_output}")
    assert final_output == expected_output, "Combined output does not match expected unmasked text!"
    print("Test 4 Passed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
