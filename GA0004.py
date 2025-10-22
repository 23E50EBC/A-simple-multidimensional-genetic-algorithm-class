import random
import matplotlib.pyplot as plt
import numpy as np
from cloudinit.analyze.show import event_parent
from matplotlib import cm  # 用于颜色映射

class MyGeneticAlgorithm:
    def __init__(self,
                 pop_size = 1, target_dict=None,
                 mutation_rate = 0.05, max_gen = 1,
                 selection_pressure_scale = 2
                 ):
        # 外部可配置的参数
        self.pop_size = pop_size  # 种群大小（96）
        #self.dim = dim  # 维度
        #self.var_ranges_down = var_ranges_down  # 每个维度的范围（x1: (-15,15), x2: (-15,15)）
        #self.var_ranges_up = var_ranges_up
        #一个更好的方式是使用一个字典来完成他们
        self.dim_renge_dict = target_dict
        #应当传入的target将符合如下格式
        #{dim1:(range_down1,range_up1),dim2:(range_down2,range_up2)...

        self.mutation_rate = mutation_rate  # 我们在下面的变异过程中使用高斯噪声模拟变异
        #高斯噪声将会根据原来的点为均值，这个值为标准差，产生高斯噪声并叠加到原有数据上
        self.max_gen = max_gen  # 最大迭代次数（50）
        self.selection_pressure_scale = selection_pressure_scale    #选择压系数
        #这个系数sps代表新的种群中只会有pop//sps这么多的个体可以成为父母，
        #譬如种群规模100,sps=2，那么就只能有100//2 = 50个对象被选择成为父母

        # 内部状态（初始化空值，后续运行时填充）
        self.individuals = {}  # 当前种群：{编号: [x1, x2, fitt]}
        self.history = []  # 每代最优解历史：[(x1, x2, fitt), ...]
        self.fitness_func = None  # 适应度函数（外部传入）
        #适应度函数应当返回一个浮点值
        self.constraint_func = None  # 约束函数b（外部传入）
        #允许用户输入一个外部函数的回调，在下面的应用中，每个子代在生成后立刻进行约束函数b的判定，
        #约束函数b应当允许接受设计的个体格式，并返回一个布尔值（但你别真写true/false），根据布尔值判定是否应当移除该子代
        #方法也特别简单，在a给出一个浮点的适应度后，b给一个布尔型的，他俩直接乘一起就得到最后的适应度


    def set_fitness_func(self, func):
        """设置适应度函数：输入x1, x2，返回fitt"""
        self.fitness_func = func

    def set_constraint_func(self, func):
        """设置约束函数：输入x1, x2，返回y"""
        self.constraint_func = func

    def _init_population(self):
        # 初始化种群
        #self.individuals = {}  # 当前种群：{编号: [x1, x2, fitt]}

        number_of_dim = len(self.dim_renge_dict)      #这条得到指定维度字典中有多少键值对
        for num in range(self.pop_size):    #每个个体
            self.individuals[num] = []      #先初始化为空列表
            for dimx in range(number_of_dim):        #每个维度
                dimx_val = self.dim_renge_dict[dimx+1]    #提取出这个维度的上下界值
                #(习惯上，我们都不把维度认为是从0开始的，但数组下标从0开始，所以这里+1）

                #注意到我们强制要求用户维度起名为1,2,3,他的值对应上下界的元组
                #因为在这里我们按照第一个，第二个这样的顺序从字典中索引每个维度
                #这意味着，你输入的顺序不重要，但必须按照这个格式，连续的输入
                dimx_down = dimx_val[0]
                dimx_up = dimx_val[1]

                #在指定的区间内随机获得一个值
                dimx_result = random.uniform(dimx_down,dimx_up)
                #这个值作为这个个体的第x个维度写入个体字典
                self.individuals[num].append(dimx_result)

        #检验一下
        print(f" def _init_population(self):,self.individuals \n {self.individuals}")

    def _evaluate(self):
        # self.individuals = {}  # 当前种群：{编号: [x1, x2, fitt]}
        min_fitt = float('inf')
        for item,val in self.individuals.items():       #对每一个个体
            self.individuals[item].append(1)    #先把fitt的那个位置初始化为1
            func_result = self.fitness_func(val)        #往外部函数传参以得到适应度评价
            #注意到，这里直接把值整个传递过去了，你在另一头注意要解包
            #在得到一个返回值后，直接追加在这个item的列表最后，由于前面初始化为1,这里直接乘法
            self.individuals[item][-1]*=func_result
            current_val = func_result

            #这里是适应度函数b，或者说不等式约束b，
            func_result_2 = self.constraint_func(val)
            self.individuals[item][-1] *= func_result_2
            current_val*=func_result_2

            #偏移量记录
            if current_val < min_fitt:
                min_fitt = current_val


        #在完成了这一步后，我们就得到了个体的fitt项，并且强制为列表的最后一个
        print(f"def _evaluate(self):,self.individuals \n {self.individuals}")
        #处理偏移
        #我们需要对每一个fitt-最小的那个fitt，这样最小的那个会被偏移到0,凡比他大的都会不小于0
        for item in self.individuals.keys():
            self.individuals[item][-1] -= min_fitt
            self.individuals[item][-1] += 10e-6     #防除0
        print(f"def _evaluate(self):,self.individuals，  处理偏移 \n {self.individuals}")

    def _selection(self):
        #我们需要首先重新构建一个编号和fitt/总fitt的列表，等到下面使用random的choice时用
        ids = list(self.individuals.keys())     #转化成列表
        #更好的选择是一个列表推导式
        fitts = []
        for num_id in ids:
            id_data = self.individuals[num_id]
            id_fitt = id_data[-1]
            #更好的选择是id_fitt = self.individuals[item][-1]
            fitts.append(id_fitt)

        total_fitt = sum(fitts)     #计算总的适应度
        #在进行下一步权重计算时，需要特殊处理
        if total_fitt == 0:
            # 所有个体适应度为0，按等概率抽样
            weights = None  # None表示等概率
        else:
            # 正常计算权重：每个个体的适应度占比
            weights = []
            for fitt in fitts:
                weights.append(fitt/total_fitt)

        print(f"def _selection(self):,weights is \n {weights}")
        #然后使用random根据权重列表，从中重新抽取出来父母
        selected_ids = random.choices(ids, weights=weights, k=self.pop_size//self.selection_pressure_scale)

        #根据被抽出来的编号，重新构建父母的字典
        selected_parents = {}   #这里提到了，由于choice可以重复选择父母，你如果用字典那重复等于没有重复，减缓收敛
        #但我的解决方案是，重新编号父母
        parents_counter = 1
        for pre_ids in selected_ids:
            val = self.individuals[pre_ids]
            selected_parents[parents_counter] = val
            parents_counter += 1

        #最后吐出去给下一步用
        print(f"myGA._selection(),return selected_parents \n {selected_parents}")
        return selected_parents

    def _crossover(self, parents):
        #首先选出父母对，这里选出所有奇数和偶数的
        odd_parent_id = []      #初始化两个列表，注意到这个列表是键的列表
        even_parent_id = []
        for index in parents.keys():
            #因为上一步对父母组重新编号了，这里直接看keys的奇数偶数就ok
            if index % 2 == 0:
                even_parent_id.append(index)
            else:
                odd_parent_id.append(index)

        print(f"def _crossover(self, parents):odd_parent_id\n{odd_parent_id}")
        print(f"def _crossover(self, parents):even_parent_id \n{even_parent_id}")

        #经过了上面一部后，这个地方会得到只有位置奇数的父母索引的列表和只有位置偶数的位置索引的列表
        #生成子代，首先需要初始化child
        childs = {}
        childs_counter = 1
        #然后，同时遍历两个列表，注意到你需要预处理保证这俩列表等长
        len_of_parents = min(len(odd_parent_id),len(even_parent_id))
        print(len_of_parents)
        for i in range(len_of_parents):
            #在父母的俩列表里找好父母
            #再获取对应的参数的值
            #注意到这个parents是这个格式的  {18: [1.5956913257385286, 6.9637495905903, 51.040039167483755],...
            par_A_id = odd_parent_id[i]     #根据顺序取出id
            par_A = parents[par_A_id]       #根据id取出数据
            par_B_id = even_parent_id[i]
            par_B = parents[par_B_id]
            for j in range(self.pop_size//len_of_parents):
                #父母越少，每个父母就应当越多产生子代，这里的计算是，总数//父母对数
                child_data = []
                for dim in range(len(self.dim_renge_dict)):     #在单个子代的每一个维度上
                    #每个维度都应该独立的进行混合
                    mix_scale = random.random()  # 这会返回一个0～1之间的数
                    mixed_dim = (1-mix_scale) * par_A[dim] + mix_scale * par_B[dim]
                    child_data.append(mixed_dim)

                #在完成了len(self.dim_renge_dict)个维度后，我们得到了一个完整的子代的数据，下一步把他加入到子代字典中
                childs[childs_counter]=child_data
                childs_counter += 1     #因为i是拿来遍历父母列表的，j是用来翻倍子代数量的，dim是一个子代的数据维度
                #最简单的给子代计数以构建字典的方式，就是再另外维护一个计数器

        print(f"def _crossover(self, parents):,childs1111\n{childs}")
        #考虑到，有可能父母挑选出来是一个奇数，譬如13个，6个odd，7个even，但上面取得是二者长度最小，故可能有的父母没被选上
        #这里做一个总长度的检测，然后直接重复最后那个每选上的父母让他自交几次把这个少的部分补上
        if len(childs) != self.pop_size:
            if len(odd_parent_id) > len(even_parent_id):
                #从多的那个集合里选亲代
                last_id = odd_parent_id[-1]
                last_par = parents[last_id]
                for j in range(self.pop_size - len(childs)):    #缺少几个补上几个
                    child_data = []
                    for dim in range(len(self.dim_renge_dict)):  # 在单个子代的每一个维度上
                        mixed_dim = last_par[dim]
                        child_data.append(mixed_dim)

                    # 在完成了len(self.dim_renge_dict)个维度后，我们得到了一个完整的子代的数据，下一步把他加入到子代字典中
                    childs[childs_counter] = child_data
                    childs_counter += 1
            else:
                # 从多的那个集合里选亲代
                last_id = even_parent_id[-1]
                last_par = parents[last_id]
                for j in range(self.pop_size - len(childs)):  # 缺少几个补上几个
                    child_data = []
                    for dim in range(len(self.dim_renge_dict)):  # 在单个子代的每一个维度上
                        mixed_dim = last_par[dim]
                        child_data.append(mixed_dim)

                    # 在完成了len(self.dim_renge_dict)个维度后，我们得到了一个完整的子代的数据，下一步把他加入到子代字典中
                    childs[childs_counter] = child_data
                    childs_counter += 1

        print(f"def _crossover(self, parents):,childs2222\n{childs}")
        return childs

    def _mutation(self, childs):
        """对每一个子代的每一个dim添加一个高斯噪声"""
        for item,val in childs.items():     #对每一个个体
            for dim in range(len(self.dim_renge_dict)):     #的每一个维度
                noise = np.random.normal(loc=0.0,scale=self.mutation_rate)      #生成高斯噪声
                #childs[item][dim]+=noise        #添加上去
                #截断，为了防止高斯噪声的偏移让这个dim离开他的定义区域
                dim_val = self.dim_renge_dict[dim + 1]  # 提取出这个维度的上下界值
                dim_down = dim_val[0]
                dim_up = dim_val[1]
                #截断操作
                childs[item][dim] = min(        #在上界和偏移后的取得最小的
                    childs[item][dim]+noise,
                    dim_up
                )
                childs[item][dim] = max(        #在下界和偏移后取得最大的
                    childs[item][dim] + noise,
                    dim_down
                )

        #处理完成的子代是新的种群
        self.individuals = childs
        print(f"def _mutation(self, childs):,childs\n{childs}")
        self.history.append(childs)

    def run(self):
        self._init_population()
        for i in range(self.max_gen):
            print(f"this is the {i}th turn")
            self._evaluate()
            my_parents = self._selection()
            my_childs = self._crossover(my_parents)
            self._mutation(my_childs)
        self._evaluate()

    def get_best_solution(self):
        """找当前评分最高的项"""
        #手动的找最大值，先初始化负无穷的适应度
        max_fitt = -float('inf')
        target = None       #以及期望的兼职对

        #获得每一个兼职对
        for key,val in self.individuals.items():
            current_fitt = val[-1]  #拿到适应度
            #比大小
            if current_fitt > max_fitt:
                max_fitt = current_fitt
                target = (key,val)

        print(target)
        return target

    def get_history(self):
        return self.history

    def get_now(self):
        return self.individuals


def fitness_A(val):
    #return val[0]**2+val[1]**2
    return np.sin(val[0]) + np.sin(val[1])

def fitness_B(val):
    if val[1] > 6:
        return 0
    else:
        return 1

if __name__ == "__main__":

    test_dict = {1:(-1.5*np.pi,1.5*np.pi),2:(-1.5*np.pi,1.5*np.pi)}

    myGA = MyGeneticAlgorithm(
        pop_size= 20,
        target_dict=test_dict,
        selection_pressure_scale=2,
        max_gen=10,
        mutation_rate=0.1
    )
    myGA.set_fitness_func(fitness_A)
    myGA.set_constraint_func(fitness_B)
    myGA.run()

    bast1 = myGA.get_best_solution()

    # --------------------------
    # 1. 准备数据
    # --------------------------
    # 生成x和y的网格数据（范围：-2π到2π，间隔0.1）
    x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    y = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)  # 转换为网格矩阵（二维数组）

    # 计算z值：z = sin(x) + sin(y)
    #z_grid = np.sin(x_grid) + np.sin(y_grid)
    z_grid = fitness_A([x_grid,y_grid])
    # 提取字典中的x和y（忽略第三个值）
    scatter_x = []
    scatter_y = []
    point_dict = myGA.get_now()
    for key in point_dict:
        x = point_dict[key][0]  # 第一个元素为x
        y = point_dict[key][1]  # 第二个元素为y
        scatter_x.append(x)
        scatter_y.append(y)
    # 转换为numpy数组（便于计算z值）
    scatter_x = np.array(scatter_x)
    scatter_y = np.array(scatter_y)
    # 计算散点对应的z值（与曲面函数一致：z = sin(x) + sin(y)）
    scatter_z = np.sin(scatter_x) + np.sin(scatter_y)
    # --------------------------
    # 2. 创建Figure和3D Axes对象（面向对象核心）
    # --------------------------
    # 注意：3D绘图需要指定projection='3d'
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})

    # --------------------------
    # 3. 绘制3D曲面（调用3D Axes的方法）
    # --------------------------
    # 绘制z = sin(x) + sin(y)的曲面，使用颜色映射表示z值大小
    surface = ax.plot_surface(
        x_grid, y_grid, z_grid,  # 网格数据（x, y, z均为二维数组）
        cmap=cm.viridis,         # 颜色映射（可选：viridis, plasma, inferno等）
        edgecolor='none',        # 关闭网格边缘线（让曲面更平滑）
        alpha=0.2                # 透明度（0-1）
    )
    # 3.2 绘制字典中的散点（面向对象：调用ax.scatter）
    scatter = ax.scatter(
        scatter_x, scatter_y, scatter_z,  # 散点的(x, y, z)
        color='red',  # 散点颜色（与曲面区分）
        s=100,  # 散点大小
        marker='o',  # 散点形状（圆形）
        edgecolor='black',  # 边缘颜色
        linewidth=1.5,  # 边缘宽度
        label='point'  # 标签（用于图例）
    )
    # --------------------------
    # 4. 设置3D图表属性（调用3D Axes的方法）
    # --------------------------
    # 设置坐标轴标签
    ax.set_xlabel('x', fontsize=12, labelpad=10)
    ax.set_ylabel('y', fontsize=12, labelpad=10)
    ax.set_zlabel('z', fontsize=12, labelpad=10)

    # 设置坐标轴范围（可选，让图像比例更协调）
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_zlim(-2, 2)  # sin(x)+sin(y)的范围是[-2, 2]

    # 设置坐标轴刻度标签（用π表示，更直观）
    ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax.set_yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])

    # 添加颜色条（表示z值与颜色的对应关系）
    cbar = fig.colorbar(
        surface,          # 关联到曲面对象
        ax=ax,            # 绑定到当前坐标轴
        shrink=0.8,       # 颜色条长度（相对于坐标轴）
        aspect=10         # 颜色条宽高比
    )
    cbar.set_label('z', fontsize=10, labelpad=10)

    # 调整视角（仰角30度，方位角45度，可根据需要修改）
    ax.view_init(elev=30, azim=45)

    # --------------------------
    # 5. 显示图表
    # --------------------------
    plt.tight_layout()
    plt.show()
