from typing import Optional, Tuple

import torch

@torch.jit.script
def extract_exponent_mantissa(t: torch.Tensor):
    if t.dtype != torch.bfloat16:
        raise ValueError("only torch.bfloat16 is supported")

    t = abs(t)
    e = torch.floor(torch.log2(t))
    m = (t/(2**e) - 1)*(2**7)
    return e, m


# This is unfortunately not scriptable
def extract_by_view(t: torch.Tensor):
    t = abs(t)
    t_int16 = t.view(torch.int16)
    e = (t_int16 & 0b0_11111111_0000000) >> 7
    m = (t_int16 & 0b0_00000000_1111111)
    return e, m


def convert_bfloat16_to_uint8(tensor_bfloat16: torch.Tensor, n_mantissa: int) -> torch.Tensor:
    """ Converts from a bfloat16 tensor (quantized to fp8 precision) to uint8 tensor.
    Cases:
        - subnormal
            - <= smallest_subnormal
            - encode as subnormal:
                - zero out exponent
                - adjust mantissa size
        - normal
            - > smallest_subnormal
            - encode as normal:
                - apply target_bias
                - mask exponent to target_exponent_mask
                - set exponent
                - set mantissa
        - extended range E4M3
            - > largest_non_extended_normal
            - mask to target_exponent_mask and set
            - mask to target_mantissa_mask and set
        - E5M2 and infinity:
            - bfloat16 FF exponent and zero mantissa
            - E5M2:
                - mask to target_exponent_mask and set
                - mask to target_mantissa_mask and set
        - nan
            - bfloat16 infinity criteria
        - in all cases: set sign bit
    """
    bfloat16_n_mantissa = 7
    bfloat16_n_exponent = 8
    bfloat16_bias = 127
    bfloat16_exponent_all_ones = 255

    n_exponent = 7 - n_mantissa 
    if n_mantissa == 2:
        bias = 15
        largest_subnormal = 2 ** -14 * 3 / 4
        largest_normal = 2 ** 15 * (1 + 3 / 4)
        largest_normal_uint8 = 0b0_11110_11
        inf_value = 0b0_11111_00
        neg_inf_value = 0b1_11111_00
        nan_value = 0b0_11111_01
        target_exponent_mask = 0b0001_1111
        target_mantissa_mask = 0b0000_0011
    elif n_mantissa == 3:
        bias = 7
        largest_subnormal = 2 ** -6 * 7 / 8
        largest_normal = 2 ** 8 * (1 + 6 / 8)
        largest_normal_uint8 = 0b0_1111_110
        inf_value = 0b0_1111_111
        neg_inf_value = 0b1_1111_111
        nan_value = 0b0_1111_111
        target_exponent_mask = 0b0000_1111
        target_mantissa_mask = 0b0000_0111
    else:
        raise Exception(f"Invalid value for n_mantissa {n_mantissa}")

    do_clamp = (largest_normal < abs(tensor_bfloat16)) & torch.isfinite(tensor_bfloat16)

    is_subnormal = (0 < abs(tensor_bfloat16)) & (abs(tensor_bfloat16) <= largest_subnormal)

    exponent, mantissa = extract_exponent_mantissa(tensor_bfloat16)
    is_negative = tensor_bfloat16 < 0

    exponent_bits = (exponent + bfloat16_bias).to(torch.int16)
    mantissa_bits = mantissa.to(torch.int16)
    mantissa_non_zero = mantissa_bits > 0

    # Reduce precision to E5M2/E4M3
    mantissa_bits >>= (bfloat16_n_mantissa - n_mantissa)

    exponent_all_ones = exponent_bits == bfloat16_exponent_all_ones

    # Explicitly compute the mantissa in the subnormal case
    mantissa = torch.where(is_subnormal,
                           abs(tensor_bfloat16) * 2 ** (bias - 1) * 2 ** n_mantissa,
                           mantissa_bits).to(torch.uint8)

    # Exponent is zero if value is subnormal
    exponent = torch.where(is_subnormal,
                           0,
                           exponent_bits).to(torch.uint8)

    exponent_all_zeros = exponent == 0

    # Apply target bias for normal and subnormal cases
    # Preserve exponent if all one or all zero
    exponent = torch.where(exponent_all_ones | exponent_all_zeros, exponent, exponent - bfloat16_bias + bias)

    if n_mantissa == 3:
        is_nan = exponent_all_ones & mantissa_non_zero
        mantissa = torch.where(is_nan, 0b01111111, mantissa)

    # Set sign, exponent and mantissa
    chopped = is_negative.to(torch.uint8)
    chopped <<= n_exponent
    chopped |= exponent & target_exponent_mask
    chopped <<= n_mantissa
    chopped |= mantissa & target_mantissa_mask

    # Calculate clamped values 
    clamped = is_negative.to(torch.uint8)
    clamped <<= (n_exponent + n_mantissa)
    clamped |= largest_normal_uint8

    chopped = torch.where(do_clamp, clamped, chopped)

    # Handle nan
    chopped = torch.where(tensor_bfloat16.isnan(), nan_value, chopped)

    # Handle infinity 
    chopped = torch.where(tensor_bfloat16.isinf(), 
                          torch.where(is_negative, neg_inf_value, inf_value),
                          chopped)
    
    return chopped

def convert_uint8_to_bfloat16(input_tensor: torch.Tensor, n_mantissa: int):
    """ Convert fp8 tensor represented int torch.uint8 to bfloat16
    - Place sign, exponent and mantissa in to bfloat16 [s][8 bits exponent][7 bits mantissa].
    - Handle fp8 special values and subnormals.
    """
    n_exponent = 7 - n_mantissa
    mantissa_divisor = 2 ** n_mantissa
    if n_mantissa == 2:
        bias = 15
        sign_mask = 0b10000000
        exponent_mask = 0b01111100
        mantissa_mask = 0b00000011
    elif n_mantissa == 3:
        bias = 7
        sign_mask = 0b10000000
        exponent_mask = 0b01111000
        mantissa_mask = 0b00000111
    else:
        raise Exception(f"Invalid value for n_mantissa {n_mantissa}")

    sign_bit = (input_tensor & sign_mask) >> 7
    is_positive = sign_bit == 0
    is_negative = sign_bit == 1
    sign = (sign_bit).to(torch.bfloat16)
    sign = (sign * 2.0 - 1.0) * -1

    exponent_bits = (input_tensor & exponent_mask) >> n_mantissa
    mantissa_bits = input_tensor & mantissa_mask

    exponent_all_zeros = exponent_bits == 0

    # exponent is signed
    exponent = exponent_bits.to(torch.int8)
    exponent = (exponent - bias).to(torch.bfloat16)

    mantissa = mantissa_bits.to(torch.bfloat16)
    mantissa = torch.where(exponent_all_zeros,
                           (mantissa / mantissa_divisor) / 2 ** (bias - 1),
                           (mantissa / mantissa_divisor + 1.0) * 2 ** exponent)
    output = mantissa * sign

    # check for special values: NaN, inf, -inf
    exponent_all_ones = exponent_bits == ((2 << (n_exponent - 1)) - 1)
    
    if n_mantissa == 2:
        mantissa_all_zeros = mantissa_bits == 0
        is_infinity = exponent_all_ones & mantissa_all_zeros
        is_nan = exponent_all_ones & ~mantissa_all_zeros
        output = torch.where(is_infinity & is_positive, float('inf'), output)
        output = torch.where(is_infinity & is_negative, float('-inf'), output)
        output = torch.where(is_nan, float('NaN'), output)
    else:
        # E4M3 doesn't have infinify
        mantissa_all_ones = mantissa_bits == ((2 << (n_mantissa - 1)) - 1)
        is_nan = exponent_all_ones & mantissa_all_ones
        output = torch.where(is_nan, float('NaN'), output)

    return output

def round_fp8(
        t: torch.Tensor,
        n_mantissa: int,
        scaling_factor: float = 1,
) -> torch.Tensor:
    if t.dtype is not torch.bfloat16:
        raise Exception('Excepted torch.bfloat16, got', t.dtype) 

    t = t * scaling_factor

    chop = convert_bfloat16_to_uint8(t, n_mantissa)
    chop_bfloat16 = convert_uint8_to_bfloat16(chop, n_mantissa)
    chop_next = chop + chop.sign().to(torch.uint8)

    chop_bfloat16 = convert_uint8_to_bfloat16(chop, n_mantissa)
    chop_next_bfloat16 = convert_uint8_to_bfloat16(chop_next, n_mantissa)

    intervals = abs(chop_bfloat16 - t)
    gap = abs(chop_bfloat16 - chop_next_bfloat16)

    # Probability of rounding down to x1
    probs_chop = intervals/gap
    
    random_numbers = torch.rand_like(probs_chop)
    chop_mask = (random_numbers > probs_chop)
    rounded = torch.where(chop_mask, chop, chop_next)

    return rounded

@torch.jit.script
def undo_round_fp8(
        fp8_tensor: torch.Tensor,
        n_mantissa: int,
        scaling_factor: float = 1,
) -> torch.Tensor:

    result = convert_uint8_to_bfloat16(fp8_tensor, n_mantissa)
    result /= scaling_factor

    return result
