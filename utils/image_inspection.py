potentially_problematic = [
    8,12,18,24,26,42,46,59,
    65,78,80,92,114,125,162,
    166,219,223,236,246,256,
    288,325,330,349,357,388,441,
    442,471,488,491,535,536,557,
    615,636,673,699,771,814,829,
    844,854,863,877,889,890,923,
    958,988
]
#check 65
#check 92
#check 125
#check 140
#check 219
#check 330
#check 388
#check 806 +
#check 844 
#check 863
#check 889

bad = [
    34,37,99,209,320,500,550,735,760,764,
    806,865
]
#check 760
might_be_wrong = [
    13, # compare with 12
    15,
    54,
    85,
    86,
    117,
    261,
    205,
    306,
    374,
    448, #check
    502, #check
    549,
    588,
    671,
    713,
    724,
    783,
    813,
    847,
    875,
    942,
    984, #check +
]

if __name__ =="__main__":
    print(len(potentially_problematic))
    print(len(bad))
    print(len(might_be_wrong))
    