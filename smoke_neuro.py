import cv2
import os
import sys
import math
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


def construct_inceptionv3onfire(x,y, training=False, enable_batch_norm=True):  

    network = input_data(shape=[None, y, x, 3])

    conv1_3_3 = conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 = conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    if enable_batch_norm:
        pool1_3_3 = batch_normalization(pool1_3_3)
    conv1_7_7 = conv_2d(pool1_3_3, 80,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    conv2_7_7 = conv_2d(conv1_7_7, 96,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    pool2_3_3= max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation='relu', name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    inception_5a_1_1 = conv_2d(inception_3a_output, 96, 1, activation='relu', name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 64, filter_size=[1,7],  activation='relu',name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1,96, filter_size=[7,1],  activation='relu',name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 64, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 96, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_3a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 96, filter_size=1, activation='relu', name='inception_5a_pool_1_1')

    
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)



    inception_7a_1_1 = conv_2d(inception_5a_output, 80, 1, activation='relu', name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_5a_output, 96, filter_size=1, activation='relu', name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[1,3],  activation='relu',name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[3,1],  activation='relu',name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_5a_output, 66, filter_size=1, activation='relu', name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[1,3],  activation='relu',name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[3,1],  activation='relu',name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 96, filter_size=1, activation='relu', name='inception_7a_pool_1_1')

    
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)



    pool5_7_7=global_avg_pool(inception_7a_output)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Detect smoke v3 (release)')
    parser.add_argument("-m", "--model_to_use", type=int, help="using model", default=3, choices={3})
    parser.add_argument('video_file', metavar='video_file', type=str, help='specify video file')
    args = parser.parse_args()

    

    print("Constructing InceptionV3")

    # if (args.model_to_use == 1): # Beta

    #     model = construct_inceptionv1onfire (224, 224, training=False)
    #     model.load(os.path.join("models/smokeV1-FireVision", "inceptiononv1onfire"),weights_only=True)

    if (args.model_to_use == 3):

        model = construct_inceptionv3onfire (224, 224, training=False)
        model.load(os.path.join("models/smokeV3-FireVision", "inceptionv3onfire"),weights_only=False)

    else:
        print("Please use v3 FireVision method")

    print("Loaded CNN network weights ...")

    rows = 224
    cols = 224

    

    windowName = "Hakaton OwlsTeam - FireVision";
    keepProcessing = True;

    video = cv2.VideoCapture(args.video_file)
    print("Loading Video")
    
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000/fps);

    while (keepProcessing):

        start_t = cv2.getTickCount();
        

        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;
        

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        output = model.predict([small_frame])

        

        if round(output[0][0]) == 1: 
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 20)
            cv2.putText(frame,'SMOKE',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
        else:
            cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 20)
            cv2.putText(frame,'OK',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);


        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        cv2.imshow(windowName, frame);

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF
        if (key == ord('e')):
            keepProcessing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)