from ALPR import ALPR

import debug_utils

alpr_model = ALPR()

print(debug_utils.get_process_mem())

for i in range(1000):
    if i % 10 == 0:
        print(i, debug_utils.get_process_mem())
    alpr_model.lp_detect("./test/test1.jpg")

print(debug_utils.get_process_mem())