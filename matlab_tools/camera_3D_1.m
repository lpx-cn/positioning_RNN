%����665 ��λ������(mm)
%���룺
%   ����- ����f�������ܶ�density���Ƹ���N���Ƶ����꣨x_WC_i,y_WC_i,z_WC_i��i=1~N,�������Z_dir��X_dir
%   ����- ���λ�ã�x,y,z��

%�����
%   ��Ƭ�ϵ����CCS���꣨x_CC_j,y_CC_j,z_CC_j��j=1~N
clc;clear all;
room_size=[6,6,5]*1000;

f=25;
density=38/1024;  %36mm�ĸй����1024������
LED_WC=[3500 , 3500 , room_size(3);
        2500 , 2500 , room_size(3);
        2500 , 3500 , room_size(3)];

Z_dir=[0,0,1]';
X_dir=[1,0,0]';

point_interval=room_size/10;
test_point=[];
train_point=[];

%%%%%%%%%%%%%%%%%%%%%%%%%% train_point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count_train=0;
for i=0:point_interval(1):room_size(1)
    for j=0:point_interval(2):room_size(2)
        for k=0:point_interval(3):(room_size(3)-point_interval(3))
            count_train=count_train+1;
            train_point(count_train,:)=[i,j,k];      
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% test_point %%%%%%%%%%%%%%%%%%%%%
count_test=0;
for i=0:point_interval(1):room_size(1)-point_interval(1)
    for j=0:point_interval(2):room_size(2)-point_interval(2)
        for k=0:point_interval(3):room_size(3)-point_interval(3)*2
            count_test=count_test+1;
            test_point(count_test,:)=[i+point_interval(1)/2,j+point_interval(2)/2,k+point_interval(3)/2];     
        end
    end
end

N_led=size(LED_WC);
N_led=N_led(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train_data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:count_train
    for j=1:N_led
        image=get_image_point_coordinate(f,density,LED_WC(j,:)',train_point(i,:)',Z_dir,X_dir);
        image_point(j,:)=[image]';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% ��֤���� %%%%%%%%%%%%%%%%%%%%%%%
    %     Pw=fun_position(LED_WC(1,:),LED_WC(2,:),LED_WC(3,:),...
    %         image_point(1,:),image_point(2,:),image_point(3,:),f);
    %     RPw=train_point(i,:);
    %     err(i)=norm(Pw-RPw);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    inputdata_train(i,:)=[f,density,image_point(:)',LED_WC(:)',Z_dir',X_dir'];
end
outputdata_train=train_point;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test_data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:count_test
    for j=1:N_led
        image=get_image_point_coordinate(f,density,LED_WC(j,:)',test_point(i,:)',Z_dir,X_dir);
        image_point(j,:)=[image]';
    end
    
    %%%%%%%%%%%%%%%%%%%%%% ��֤���� %%%%%%%%%%%%%%%%%%%%%%
    %         Pw=fun_position(LED_WC(1,:),LED_WC(2,:),LED_WC(3,:),...
    %             image_point(1,:),image_point(2,:),image_point(3,:),f);
    %         RPw=test_point(i,:);
    %         err(i)=norm(Pw-RPw);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    inputdata_test(i,:)=[f,density,image_point(:)',LED_WC(:)',Z_dir',X_dir'];
end
outputdata_test=test_point;

csvwrite('train_features.csv',inputdata_train);
csvwrite('train_labels.csv',outputdata_train);
csvwrite('test_features.csv',inputdata_test);
csvwrite('test_labels.csv',outputdata_test);

%%%%%%%%%%%%%%%%%%%%%%%%% ��ͼ ��֤ ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:100:count_train
%     for j=1:N_led
%         image=get_image_point_coordinate(f,density,LED_WC(j,:)',train_point(i,:)',Z_dir,X_dir);
%         image_point(j,:)=[image]';
%     end
%     plotwcs(f,LED_WC,image_point,train_point(i,:)',Z_dir,X_dir)
% end
% for i=1:100:count_test
%     for j=1:N_led
%         image=get_image_point_coordinate(f,density,LED_WC(j,:)',test_point(i,:)',Z_dir,X_dir);
%         image_point(j,:)=[image]';
%     end
%     plotwcs(f,LED_WC,image_point,test_point(i,:)',Z_dir,X_dir)
% end

