import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import phyre
import random
import numpy as np
from tqdm import tqdm

def T2_plot(df_value, df_xy, template_id, task_id):

    q1 = df_value.shape[1]
    r1 = df_value.shape[0]
    print(r1, q1)
    X = df_xy[:, 0]
    Y = df_xy[:, 1]
    Z = df_value[:, 0]


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X', fontsize=10, color='black')
    ax.set_ylabel('Y', fontsize=10, color='black')
    ax.set_zlabel('Score', fontsize=10, color='black')
    print(X.shape, Y.shape, Z.shape)
    ax.plot_trisurf(X, Y, Z, cmap=plt.get_cmap('jet'), linewidth=0.1)
    plt.savefig(f'./maps/map_{template_id}_{task_id}.png')


# draw initial scene
for start_id in tqdm(range(25)):
    # start_id = 0
    end_id = start_id + 1
    random.seed(0)
    np.random.seed(0)
    protocal, fold_id = 'within', 0
    print(f'visualizing protocal {protocal} and fold {fold_id}')

    # setup the PHYRE evaluation split
    eval_setup = f'ball_{protocal}_template'
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    _, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
    candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
    test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
    task_id = 0
    test_list = test_list[task_id:task_id+1]
    simulator = phyre.initialize_simulator(test_list, action_tier)
    initial_scene = simulator.initial_scenes[0]
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(phyre.observations_to_float_rgb(initial_scene))
    # plt.show()
    plt.savefig(f'./maps/initial_{start_id}_{task_id}.png')

    # draw heatmap
    save_path = f'./data_map/map_{start_id}_{task_id}.hkl'
    with open(save_path, 'rb') as f:
        distr_map = pickle.load(f)
    print(distr_map.shape)

    T2_plot(distr_map[:,3:], distr_map[:,0:2], template_id=start_id, task_id=task_id)


# import pickle
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import phyre
# import random
# import numpy as np
# from tqdm import tqdm
#
# def T2_plot(df_value, df_xy, fig):
#
#     q1 = df_value.shape[1]
#     r1 = df_value.shape[0]
#     print(r1, q1)
#     X = df_xy[:, 0]
#     Y = df_xy[:, 1]
#     Z = df_value[:, 0]
#
#     ax = Axes3D(fig)
#     ax.set_xlabel('X', fontsize=10, color='black')
#     ax.set_ylabel('Y', fontsize=10, color='black')
#     ax.set_zlabel('Score', fontsize=10, color='black')
#     print(X.shape, Y.shape, Z.shape)
#     ax.plot_trisurf(X, Y, Z, cmap=plt.get_cmap('jet'), linewidth=0.1)
#
#
# # draw initial scene
# for start_id in tqdm(range(4,5)):
#     # start_id = 0
#     end_id = start_id + 1
#     random.seed(0)
#     np.random.seed(0)
#     protocal, fold_id = 'within', 0
#     print(f'visualizing protocal {protocal} and fold {fold_id}')
#
#     # setup the PHYRE evaluation split
#     eval_setup = f'ball_{protocal}_template'
#     action_tier = phyre.eval_setup_to_action_tier(eval_setup)
#     _, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
#     candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
#     test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
#     task_id = 0
#     test_list = test_list[task_id:task_id+1]
#     simulator = phyre.initialize_simulator(test_list, action_tier)
#
#     # draw heatmap
#     save_path = f'./data_map/map_{start_id}_{task_id}.hkl'
#     with open(save_path, 'rb') as f:
#         distr_map = pickle.load(f)
#     print(distr_map.shape)
#
#     fig = plt.figure()
#
#     T2_plot(distr_map[:,3:], distr_map[:,0:2], fig=fig)
#
#     initial_scene = simulator.initial_scenes[0]
#     fig.add_subplot(5, 5, 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(phyre.observations_to_float_rgb(initial_scene))
#     plt.show()
#     # plt.savefig(f'./{start_id}_{task_id}.png')