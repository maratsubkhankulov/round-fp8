import pytest
from functools import reduce

from .round import *

e5m2 = {
    "0":                  0b0_00000_00,
    "smallest_subnormal": 0b0_00000_01,
    "largest_subnormal":  0b0_00000_11,
    "smallest_normal":    0b0_00001_00,
    "largest_normal":     0b0_11110_11,
    "inf":                0b0_11111_00,
    "nan1":               0b0_11111_01,
    "nan2":               0b0_11111_10,
    "nan3":               0b0_11111_11,
    "-0":                 0b1_00000_00,
    "-smallest_subnormal":0b1_00000_01,
    "-largest_subnormal": 0b1_00000_11,
    "-smallest_normal":   0b1_00001_00,
    "-largest_normal":    0b1_11110_11,
    "-inf":               0b1_11111_00,
    "nan4":               0b1_11111_01,
    "nan5":               0b1_11111_10,
    "nan6":               0b1_11111_11,
}

e4m3 = {
    "0":                  0b0_0000_000,
    "smallest_subnormal": 0b0_0000_001,
    "largest_subnormal":  0b0_0000_111,
    "smallest_normal":    0b0_0001_000,
    "largest_normal":     0b0_1110_111,
    "largest_normal_ext": 0b0_1111_110,
    "nan1":               0b0_1111_111,
    "-0":                 0b1_0000_000,
    "-smallest_subnormal":0b1_0000_001,
    "-largest_subnormal": 0b1_0000_111,
    "-smallest_normal":   0b1_0001_000,
    "-largest_normal":    0b1_1110_111,
    "-largest_normal_ext":0b1_1111_110,
    "nan2":               0b1_1111_111,
}

nan_values = {
    2: {
        e5m2['nan1'],
        e5m2['nan2'],
        e5m2['nan3'],
        e5m2['nan4'],
        e5m2['nan5'],
        e5m2['nan6'],
    },
    3: {
        e4m3['nan1'],
        e4m3['nan2'],
    }
}


@pytest.mark.parametrize("test_case, n_mantissa",
[
    ("e5m2", 2),
    ("e4m3", 3),
])
def test_convert(test_case, n_mantissa):
    start_value = 0b0000_0000
    end_value   = 0b1111_1111

    for num in range(start_value, end_value + 1):
        original_tensor = torch.tensor(num, dtype=torch.uint8)

        bfloat16_tensor = convert_uint8_to_bfloat16(original_tensor, n_mantissa)

        chopped = convert_bfloat16_to_uint8(bfloat16_tensor, n_mantissa)

        if torch.isnan(bfloat16_tensor):
            assert num in nan_values[n_mantissa]
            continue
    
        if bfloat16_tensor == 0:
            assert chopped == 0
            continue

        assert original_tensor == chopped


@pytest.mark.parametrize("test_case, n_mantissa, start_value, end_value",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-largest_normal"], e5m2["-0"]),
    ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    ("e4m3 negative", 3, e4m3["-largest_normal_ext"], e4m3["-0"]),
])
def test_round(test_case: str, n_mantissa: int, start_value: int, end_value):
    prev = start_value
    for curr in range(start_value + 1, end_value + 1):
        prev_tensor = torch.tensor(prev, dtype=torch.uint8)
        curr_tensor = torch.tensor(curr, dtype=torch.uint8)
        prev_ = convert_uint8_to_bfloat16(prev_tensor, n_mantissa)
        curr_ = convert_uint8_to_bfloat16(curr_tensor, n_mantissa)

        # mid point is higher precision than curr and prev
        mid_point = (curr_ + prev_)/2

        # test that we correctly round up and down
        chopped = convert_bfloat16_to_uint8(mid_point, n_mantissa)

        assert chopped == prev_tensor

        prev = curr


@pytest.mark.parametrize("test_case, n_mantissa, start_value, end_value",
[
    ("e5m2 positive", 2, e5m2["0"], e5m2["largest_normal"]),
    ("e5m2 negative", 2, e5m2["-largest_normal"], e5m2["-0"]),
    ("e4m3 positive", 3, e4m3["0"], e4m3["largest_normal_ext"]),
    ("e4m3 negative", 3, e4m3["-largest_normal_ext"], e4m3["-0"]),
])
def test_stochastic_rounding(test_case, n_mantissa, start_value, end_value):
    scaling_factor = 1

    prev = start_value
    for curr in range(start_value + 1, end_value + 1):
        prev_tensor = torch.tensor(prev, dtype=torch.uint8)
        curr_tensor = torch.tensor(curr, dtype=torch.uint8)
        prev_ = convert_uint8_to_bfloat16(prev_tensor, n_mantissa)
        curr_ = convert_uint8_to_bfloat16(curr_tensor, n_mantissa)

        # mid point is higher precision than curr and prev
        mid_point = (curr_ + prev_)/2

        # test that we correctly round down
        fp8_rounded_tensor = round_fp8(
            mid_point,
            n_mantissa,
            scaling_factor
        )
        assert prev_tensor == fp8_rounded_tensor or curr_tensor == fp8_rounded_tensor

        prev = curr


@pytest.mark.parametrize("test_case, n_mantissa, num, offset, expected",
[
    ("e5m2 clamp if > largest positive", 2, e5m2["largest_normal"], 1, e5m2["largest_normal"]),
    ("e5m2 clamp if < smallest negative", 2, e5m2["-largest_normal"], -1, e5m2["-largest_normal"]),
    ("e4m3 clamp if > largest positive", 3, e4m3["largest_normal"], 1, e4m3["largest_normal"]),
    ("e4m3 clamp if < smallest negative", 3, e4m3["-largest_normal"], -1, e4m3["-largest_normal"]),
])
def test_round_special(test_case: str, n_mantissa, num, offset, expected):
    fp8_tensor = torch.tensor(num, dtype=torch.uint8)
    expected_fp8 = torch.tensor(expected, dtype=torch.uint8)
    bfloat16_tensor = convert_uint8_to_bfloat16(fp8_tensor, n_mantissa) + offset

    chopped_fp8 = convert_bfloat16_to_uint8(bfloat16_tensor, n_mantissa)

    assert chopped_fp8 == expected_fp8

@pytest.mark.parametrize("n_mantissa", [ 2, 3 ])
@pytest.mark.parametrize("mean", [ -1.0, 1.0 ])
def test_avg(n_mantissa, mean):
    for i in range(10):
        input = torch.randn((1024, 1024), dtype=torch.bfloat16) + mean
        fp8 = round_fp8(input, n_mantissa)
        output = undo_round_fp8(fp8, n_mantissa)

        assert torch.allclose(input.mean(), output.mean(), rtol=1e-02)
