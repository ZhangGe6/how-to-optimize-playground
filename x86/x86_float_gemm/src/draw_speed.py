import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.figure(figsize=(8, 4.8))

peak_GFlops_ps = 60 # a tempory one here

plt.ylim((0, peak_GFlops_ps))
plt.xlabel('matirx size')
plt.ylabel('GFlops')
plt.title('x86 float32 GEMM')

methods = [
            'MMult_base', 'MMult_optim1_1', 'MMult_optim2_1',
            'MMult_optim3_1', 'MMult_optim3_4', 'MMult_optim3_5',
            'MMult_optim4_1', 'MMult_optim4_2', 
            'MMult_optim5_1',
            'MMult_optim6_2', 'MMult_optim6_3',
            'MMult_optim7_2', 'MMult_optim7_3',
            'MMult_optim8_2',
            # 'MMult_optim9_1',
        ]

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_color_names = [name for i, (hsv, name) in enumerate(by_hsv) if i % 20 == 0] \
                   + [name for i, (hsv, name) in enumerate(by_hsv) if i % 30 == 0] \
                   + [name for i, (hsv, name) in enumerate(by_hsv) if i % 40 == 0]

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

# plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
plt.tight_layout()
plt.savefig('../res/' + 'cur_all' + '.png')