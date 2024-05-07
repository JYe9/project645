import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh

Hip_index = read_bvh.joint_index['hip']

Seq_len = 100
Hidden_size = 1024
Joints_num = 43 #57
Condition_num = 5
Groundtruth_num = 5
In_frame_size = Joints_num * 3 #+ 3  # Add 3 for the hip position

# print(f"Joints_num: {Joints_num}")
# print(f"In_frame_size: {In_frame_size}")

class acLSTM(nn.Module):
    def __init__(self, in_frame_size, hidden_size=1024, out_frame_size=171):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)  # param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    # output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        # c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        return ([h0, h1, h2], [c0, c1, c2])

    # in_frame b*In_frame_size
    # vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    # out_frame b*In_frame_size
    # vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)  # out b*150
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]

        return (out_frame, vec_h_new, vec_c_new)

    # output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        gt_lst = np.ones((100, groundtruth_num))
        con_lst = np.zeros((100, condition_num))
        lst = np.concatenate((gt_lst, con_lst), 1).reshape(-1)
        return lst[0:seq_len]

    # in cuda tensor real_seq: b*seq_len*frame_size
    # out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):

        batch = real_seq.size()[0]
        seq_len = real_seq.size()[1]

        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)

        # initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, 1))).cuda())

        out_frame = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.out_frame_size))).cuda())

        for i in range(seq_len):

            if (condition_lst[i] == 1):  ##input groundtruth frame
                in_frame = real_seq[:, i]
            else:
                in_frame = out_frame

            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)

            out_seq = torch.cat((out_seq, out_frame), 1)

        return out_seq[:, 1: out_seq.size()[1]]

    # cuda tensor out_seq batch*(seq_len*frame_size)
    # cuda tensor groundtruth_seq batch*(seq_len*frame_size)
    def calculate_loss(self, out_seq, groundtruth_seq):

        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


# numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False,
                       save_bvh_motion=True):
    # Extract hip_pos and euler_data
    hip_pos = real_seq_np[:, :, :3]  # assume first 3 values are hip position
    euler_data = real_seq_np[:, :, 3:]  # assume remaining values are euler angles

    # Prepare input and ground truth data
    in_real_seq = torch.autograd.Variable(torch.FloatTensor(euler_data[:, :-1].tolist()).cuda())
    predict_groundtruth_seq = torch.autograd.Variable(torch.FloatTensor(euler_data[:, 1:].tolist())).cuda().view(
        euler_data.shape[0], -1)

    # Add hip_pos to the input and ground truth data
    hip_pos_in = torch.autograd.Variable(torch.FloatTensor(hip_pos[:, :-1].tolist()).cuda())
    hip_pos_out = torch.autograd.Variable(torch.FloatTensor(hip_pos[:, 1:].tolist()).cuda())
    in_real_seq = torch.cat((hip_pos_in, in_real_seq), dim=2)

    # Reshape hip_pos_out to match the shape of predict_groundtruth_seq
    hip_pos_out = hip_pos_out.view(hip_pos_out.shape[0], -1)
    predict_groundtruth_seq = torch.cat((hip_pos_out, predict_groundtruth_seq), dim=1)

    # Forward pass
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)

    optimizer.zero_grad()

    # Calculate loss
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq)

    loss.backward()

    optimizer.step()

    if (print_loss == True):
        print("###########" + "iter %07d" % iteration + "######################")
        print(loss)
        print("loss: " + str(loss.detach().cpu().numpy()))

    if (save_bvh_motion == True):
        # Save the first motion sequence in the batch
        gt_seq = np.array(predict_groundtruth_seq[0].data.tolist()).reshape(-1, In_frame_size)
        out_seq = np.array(predict_seq[0].data.tolist()).reshape(-1, In_frame_size)

        # Write euler angles to bvh
        read_bvh.write_euler_to_bvh(save_dance_folder + "%07d" % iteration + "_gt.bvh", skeleton, non_end_bones, gt_seq)
        read_bvh.write_euler_to_bvh(save_dance_folder + "%07d" % iteration + "_out.bvh", skeleton, non_end_bones,
                                    out_seq)

# input a list of dances [dance1, dance2, dance3]
# return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        # length=len(dance)/100
        length = 10
        if (length < 1):
            length = 1
        len_lst = len_lst + [length]

    index_lst = []
    index = 0
    for length in len_lst:
        for i in range(length):
            index_lst = index_lst + [index]
        index = index + 1
    return index_lst


# input dance_folder name
# output a list of dances.
def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []
    for dance_file in dance_files:
        print("load " + dance_file)
        dance = np.load(dance_folder + dance_file)
        assert dance.shape[1] == In_frame_size, f"Expected dance to have size {In_frame_size} in dimension 1, but got {dance.shape[1]}"
        print("frame number: " + str(dance.shape[0]))
        dances = dances + [dance]
    return dances


# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, hidden_size=1024, total_iter=500000):
    seq_len = seq_len + 2
    torch.cuda.set_device(0)

    model = acLSTM(in_frame_size=In_frame_size, hidden_size=hidden_size, out_frame_size=In_frame_size)

    if (read_weight_path != ""):
        model.load_state_dict(torch.load(read_weight_path))

    model.cuda()
    # model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

    model.train()

    # dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst = get_dance_len_lst(dances)
    random_range = len(dance_len_lst)

    speed = frame_rate / 30  # we train the network with frame rate of 30

    for iteration in range(total_iter):
        # get a batch of dances
        dance_batch = []
        for b in range(batch):
            # randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0, random_range)]
            dance = dances[dance_id].copy()
            dance_len = dance.shape[0]

            start_id = random.randint(10,
                                      dance_len - seq_len * speed - 10)  # the first and last several frames are sometimes noisy.
            sample_seq = []
            for i in range(seq_len):
                sample_seq = sample_seq + [dance[int(i * speed + start_id)]]

            # augment the direction and position of the dance
            T = [0.1 * (random.random() - 0.5), 0.0, 0.1 * (random.random() - 0.5)]
            R = [0, 1, 0, (random.random() - 0.5) * np.pi * 2]
            sample_seq_augmented = read_bvh.augment_train_data(sample_seq, T, R)
            dance_batch = dance_batch + [sample_seq_augmented]

        dance_batch_np = np.array(dance_batch)

        print_loss = False
        save_bvh_motion = False
        if (iteration % 1 == 0):
            print_loss = True
        if (iteration % 1000 == 0):
            save_bvh_motion = True

        assert dance_batch_np.shape[2] == In_frame_size, f"Expected dance_batch_np to have size {In_frame_size} in dimension 2, but got {dance_batch_np.shape[2]}"
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss,
                           save_bvh_motion)
        # end=time.time()
        # print end-start
        if (iteration % 1000 == 0):
            path = write_weight_folder + "%07d" % iteration + ".weight"
            torch.save(model.state_dict(), path)


read_weight_path = ""
write_weight_folder = "../train_weight_aclstm_martial_euler/"
write_bvh_motion_folder = "../train_tmp_bvh_aclstm_martial_euler/"
dances_folder = "../train_data_euler/martial/"
dance_frame_rate = 120
batch = 32
hidden_size = 1024

if not os.path.exists(write_weight_folder):
    os.makedirs(write_weight_folder)
if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

dances = load_dances(dances_folder)

skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy("../train_data_bvh/standard.bvh")

train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder,
      write_bvh_motion_folder, hidden_size, total_iter=10000)
