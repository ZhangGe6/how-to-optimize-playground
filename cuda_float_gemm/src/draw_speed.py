import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# change it according to our machine
peak_GFlops_ps = 6000 # a tempory one here

plt.ylim((0, peak_GFlops_ps))
plt.xlabel('matirx size')
plt.ylabel('GFlops')

methods = [
        'MMul_benchmark',
        'MMul_base', 'MMul_optim1_1', 'MMul_optim2_1', 'MMul_optim3_1', 'MMul_optim3_2', 'MMul_optim3_3', 'MMul_optim3_5', 'MMul_optim4_1', 'MMul_optim4_2',
        'MMul_optim6_1'
        #    'MMul_optim1_1', 'MMul_optim1_2', 'MMul_optim1_3', 'MMul_optim1_4',
        #    'MMul_optim2_0', 'MMul_optim2_1', 'MMul_optim2_2', 'MMul_optim2_3',
        #    'MMul_optim3_1', 'MMul_optim3_2', 'MMul_optim3_3', 'MMul_optim3_4', 'MMul_optim3_5', 'MMul_optim3_6', 'MMul_optim3_7',
        # 'MMul_optim3_2', 'MMul_optim4_1', 'MMul_optim4_2', 'MMul_optim4_3', 'MMul_optim4_4', 'MMul_optim4_5', 
        # 'MMul_optim3_2', 'MMul_optim5_1', 'MMul_optim5_2',

        # 'MMul_base', 'MMul_optim3_2', 'MMul_optim6_1', 'MMul_optim6_2', 'MMul_optim6_3', 'MMul_optim6_4', 'MMul_optim6_5', 'MMul_optim6_6', 'MMul_optim6_7',
        # 'MMul_base', 'MMul_optim6_1', 'MMul_optim6_6', 'MMul_optim7_1'
        # 'MMul_base',  'MMul_optim8_0', 'MMul_optim8_1', 'MMul_optim1_1', 'MMul_optim8_2', 'MMul_optim7_1', 'MMul_optim8_3', 'MMul_optim8_4', 'MMul_optim8_5', 'MMul_optim8_7',

        # 'MMul_base', 'MMul_optim1_1', 'MMul_optim9_1', 'MMul_optim9_2', 'MMul_optim9_3'
        # 'MMul_base', 'MMul_optim8_7', 'MMul_optim9_4', 'MMul_optim9_5', 'MMul_optim9_6', 'MMul_optim9_7', 'MMul_optim9_8', 'MMul_optim9_9'

        ]

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_color_names = [name for i, (hsv, name) in enumerate(by_hsv) if i % 16 == 0] + [name for i, (hsv, name) in enumerate(by_hsv) if i % 20 == 0] + [name for i, (hsv, name) in enumerate(by_hsv) if i % 30 == 0]

for i, method in enumerate(methods):
    resfile = '../res/' + method + '.txt'
    msize_set = []
    GF_ps_set = []
    with open(resfile, 'r') as f:
        for line in f.readlines():
            [msize, Gflops_per_second, max_diff] = line.split()
            msize_set.append(int(msize))
            GF_ps_set.append(float(Gflops_per_second))

            assert float(max_diff) == 0

    plt.plot(msize_set, GF_ps_set, color=sorted_color_names[i], marker="*", label=method)

plt.legend()
plt.savefig('../res/' + 'cur_all' + '.png')