import pickle as pkl
import numpy as np
import torch
import os, time
from PIL import Image
import matplotlib.pyplot as plt
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import socket

# is_gpu = torch.cuda.is_available() 

def test_acc(net, test_loader, Target=True):
    is_gpu = torch.cuda.is_available()     
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        if is_gpu:
            images,labels = images.cuda(),labels.cuda()
        if Target==True:
            outputs = net(Variable(images))
        else :
            outputs = net(Variable(images),"source")
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        (100.0 * correct) / total))
    net.train()
    
def train_mnistm_only(net,train_loader,test_loader,optimizer,criterion,epochs,Target_check=True):
    is_gpu = torch.cuda.is_available() 
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        cudnn.benchmark = True
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs,"target")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.data[0]
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
#                 break
                test_acc(net,test_loader,Target_check)
    print('Finished Training')
    
def train_mnist_only(net,train_loader,test_loader,optimizer,criterion,epochs, Target_check=True):
    cudnn.benchmark = True
    is_gpu = torch.cuda.is_available() 
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
#             net.train()
            # get the inputs
    
            inputs, labels = data
            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()


            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs,"source")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            p_p = 200
            if i % p_p == (p_p-1):    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / p_p))
                running_loss = 0.0
#                 break
                test_acc(net,test_loader, Target_check)

    print('Finished Training')

def test_acc_joint(net, test_loader,early_stop=None):
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    net.eval()
    start_time = time.time()
    for i,data in enumerate(test_loader):
        
        data1, data2 = data
        data1_img,data1_labels = data1
        data2_img,data2_labels = data2
        # if is_gpu:
        data1_img,data1_labels = data1_img.cuda(net.rgpu[0]),data1_labels.cuda(net.rgpu[0])
        data2_img,data2_labels = data2_img.cuda(net.rgpu[0]),data2_labels.cuda(net.rgpu[0])

        # wrap them in Variable
        data1_img,data2_img = Variable(data1_img), Variable(data2_img)
        output1 = net(data1_img,"source")
        _, predicted1 = torch.max(output1.data, 1)
        total1 += data1_labels.size(0)
        correct1 += (predicted1 == data1_labels).sum()
        
        output2 = net(data2_img,"target")
        _, predicted2 = torch.max(output2.data, 1)
        total2 += data2_labels.size(0)
        correct2 += (predicted2 == data2_labels).sum()
        if (early_stop is not None)&((i%early_stop) == (early_stop-1)):    
                break
    acc_source = (100.0 * correct1) / total1
    acc_target = (100.0 * correct2) / total2
    end_time = time.time()
    test_time = end_time - start_time
    print ('Test Statistics - time: %.2f' % (test_time))
    print('Accuracy of the network on the source test images: %.2f %%' % (acc_source))
    print('Accuracy of the network on the target test images: %.2f %%' % (acc_target))
    net.train()
    return acc_source,acc_target
        
#trainer for joint dataset of mnist and mnistm

def train_mnist_joint(net, train_loader, test_loader, optimizer, criterion, epochs, save_model_file="classification/joint_dataset", early_stop=None, schedule=None):
    # cudnn.benchmark = True
    save_model_file = os.path.join('../logs', save_model_file +"_batch_size_"+str(train_loader.batch_size) +"_"+ datetime.datetime.now().strftime('%b%d_%H-%M-%S')+ '_'+socket.gethostname())
    best_source_acc = 0.0
    best_target_acc = 0.0
    print ("Starting to train the classifier on both the datasets")
    start_time = time.time()
    for epoch in range(epochs): 
        if schedule is not None and epoch in [np.floor(s*epochs) for s in schedule]:
            print("Scheduling the rate from %f to %f"%(optimizer.param_groups[0]['lr'],optimizer.param_groups[0]['lr']/10))
            optimizer.param_groups[0]['lr'] /= 10
        loss_list = []
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            net.train()
            data1, data2 = data
            data1_img,data1_labels = data1
            data2_img,data2_labels = data2
            # if is_gpu:
            data1_img,data1_labels = data1_img.cuda(net.rgpu[0]), data1_labels.cuda(net.rgpu[0])
            data2_img,data2_labels = data2_img.cuda(net.rgpu[0]), data2_labels.cuda(net.rgpu[0])
            # wrap them in Variable
            data1_img,data1_labels = Variable(data1_img), Variable(data1_labels)
            data2_img,data2_labels = Variable(data2_img), Variable(data2_labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output1 = net(data1_img,"source")
            loss1 = criterion(output1, data1_labels)
            output2 = net(data2_img,"target")
            loss2= criterion(output2, data2_labels)
            loss=loss1+loss2
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()

            if (early_stop is not None)&((i % early_stop) == (early_stop-1)):    
                break
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        avg_loss = torch.mean(torch.FloatTensor(loss_list))
        print ('[%d/%d] Statistics - per epoch time: %.2f,classification loss: %.3f' % ((epoch + 1), epochs, per_epoch_ptime, avg_loss))
        source_acc,target_acc = test_acc_joint(net,test_loader,early_stop)
        if (epoch >=(epochs/4)):
            if source_acc>best_source_acc:
                best_source_acc = source_acc
            if target_acc>best_target_acc:
                best_target_acc = target_acc
                best_model = net
    end_time = time.time()
    tot_time = end_time - start_time
    print ('Total time for training: %.2f' % (tot_time))
    print("Best accuracy on source data:%.2f"%(best_source_acc))
    print("Best accuracy on target data:%.2f"%(best_target_acc))
    torch.save(best_model.state_dict(), save_model_file)
    print ('Finished Training')
    
#trainer for vae-gan model

def train_vae_gan(G, D, E, T, train_loader, test_loader, opt_G, opt_D, opt_E, opt_T, crit_GAN, crit_prior, crit_Task, crit_cnstrnt, epochs, model_file="domain_adapt/vae_gan", early_stop=None, schedule=None):
    # cudnn.benchmark = True
    best_source_acc = 0.0
    best_target_acc = 0.0
    # results save folder
    save_model_folder = os.path.join('../logs', model_file+"_batch_size_"+str(train_loader.batch_size) +"_"+ datetime.datetime.now().strftime('%b%d_%H-%M-%S')+ '_'+socket.gethostname())
    os.mkdir(save_model_folder)
    gan_results_folder = os.path.join(save_model_folder,'MNISTM_DCGAN_results')
    os.mkdir(gan_results_folder)
    gan_results_folder_random = os.path.join(gan_results_folder,"Random_results")
    os.mkdir(gan_results_folder_random)
    gan_results_folder_fixed = os.path.join(gan_results_folder,"Fixed_results")
    os.mkdir(gan_results_folder_fixed)
    save_model_file = os.path.join(save_model_folder,"best_model")
    
    print ("Starting to train the classifier on both the datasets")
    start_time = time.time()
    for epoch in range(epochs): 
        if schedule is not None and epoch in [np.floor(s*epochs) for s in schedule]:
            print("Scheduling the rate from %f to %f"%(optimizer.param_groups[0]['lr'],optimizer.param_groups[0]['lr']/10))
            optimizer.param_groups[0]['lr'] /= 10
        G_loss_list = []
        D_loss_list = []
        E_loss_list = []
        T_loss_list = []
        
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            data1, data2 = data
            data1_img,data1_labels = data1
            data2_img,data2_labels = data2
            batch_size = data2_img.size()[0]
            #lets create labels for GAN
            y_real = torch.ones(batch_size)
            y_fake = torch.zeros(batch_size)
            # wrap them in Variable
            data2_img,data2_labels = data2_img.cuda(net.rgpu[0]), data2_labels.cuda(net.rgpu[0])
            data2_img,data2_labels = Variable(data2_img), Variable(data2_labels)
            y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())
            # Discriminator Training
            #real data through discriminator
            opt_D.zero_grad()
            D_result = D(data2_img).squeeze()
            D_real_loss = crit_GAN(D_result, y_real)
            #sampling from normal distribution
            z_ = torch.randn((batch_size, G.z)).view(-1, G.z, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            D_fake_loss = crit_GAN(D_result, y_fake)
            #conditioned generation
            data1_img,data1_labels = data1_img.cuda(net.rgpu[0]), data1_labels.cuda(net.rgpu[0])
            data1_img,data1_labels = Variable(data1_img), Variable(data1_labels)
            E_result = E(data1_img,"source")
            G_result_cond = G(E_result[2])
            D_result = D(G_result_cond)
            D_fake_loss_cond = crit_GAN(D_result, y_fake)
            #total GAN loss 
            D_train_loss = D_real_loss + D_fake_loss + D_fake_loss_cond

            D_train_loss.backward()
            opt_D.step()

            # train generator G
            opt_G.zero_grad()
            #sampling from normal 
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_fake_loss = crit_GAN(D_result, y_real)
            #conditioned generation
            D_result = D(G_result_cond)
            G_fake_loss_cond = crit_GAN(D_result, y_real)
            
            G_gan_loss = G_fake_loss_cond + G_fake_loss
            #Generating latent rep. distribution
            E_result_target = E(G_result_cond)
            T_result = T(E_result_target[2])
            
            #KL loss of these two distributions
            # G_train_loss = G_gan_loss + G_kl_loss
            G_train_loss.backward()
            opt_G.step()

            #training Encoder
            E_kl_loss = G_kl_loss
            #T_result = T(E_result)
            #
            output1 = net(data1_img,"source")
            loss1 = criterion(output1, data1_labels)
            output2 = net(data2_img,"target")
            loss2= criterion(output2, data2_labels)
            loss=loss1+loss2
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()

            if (early_stop is not None)&((i % early_stop) == (early_stop-1)):    
                break
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        avg_loss = torch.mean(torch.FloatTensor(loss_list))
        print ('[%d/%d] Statistics - per epoch time: %.2f,classification loss: %.3f' % ((epoch + 1), epochs, per_epoch_ptime, avg_loss))
        source_acc,target_acc = test_acc_joint(net,test_loader,early_stop)
        if (epoch >=(epochs/4)):
            if source_acc>best_source_acc:
                best_source_acc = source_acc
            if target_acc>best_target_acc:
                best_target_acc = target_acc
                best_model = net
    end_time = time.time()
    tot_time = end_time - start_time
    print ('Total time for training: %.2f' % (tot_time))
    print("Best accuracy on source data:%.2f"%(best_source_acc))
    print("Best accuracy on target data:%.2f"%(best_target_acc))
    torch.save(best_model.state_dict(), save_model_file)
    print ('Finished Training')
