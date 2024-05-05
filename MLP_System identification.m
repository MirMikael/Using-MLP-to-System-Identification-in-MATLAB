% Created by Mir Mikael
% For quastions: MirMikael.github.io
%%
clc
clear
close all
%%
u=rand(1,1000);
y=zeros(1,1000);
for k=0:999
    if k==0
        y(1)=0;
    else if k==1
            y(2)=u(1);
        else if k==2
                y(3)=u(2)/1+y(1)^2;
            else k>2;
                y(k+1)=((y(k)*y(k-1)*y(k-2)*u(k-1)*(y(k-2))+u(k))/(1+(y(k-1))^2+(y(k-2))^3));
                x(:,k+1)=[y(k);y(k-1);y(k-2);u(k);u(k-1)];
            end
        end
    end
end
%%
N=numel(y);
%num of train samples
ns = ceil(0.7*N);
%num of validation samples
nv = ceil(0.15*N);
%num of test samples
nt = N-(ns+nv);

% network architecture
ni=5;%network input
no=1;%network output
nh1=11;%num of hidden layer nerouns (first hidden layer)
eta=0.01;

Tr = [];Ve=[];Tc=[];


epoch_Num = 150;

mean_square_error=[];
D=y(1:ns);
p=x(:,1:ns);
pv=x(:,ns+1:ns+nv);
Dv=y(ns+1:ns+nv);
pt=x(:,ns+nv+1:end);
Dt=y(ns+nv+1:end);

% weights & biases initialization
w1=rand(nh1,ni)*2-1;
w2=rand(no,nh1)*2-1;

b1=rand(nh1,1)*2-1;
b2=rand(no,1)*2-1;
w_1=[];w_2=[];w_3=[];w2_2=[];


for index_of_epochs = 1 : epoch_Num
    error=[];
    for i=1:ns
        %% feedforward
        out0=x(:,i);
        net1=w1*out0;
        out1=tanh(net1);
        net2=w2*out1;
        out2=purelin(net2);
        Y(i)=out2;
        error = [error,D(i)-out2];% e = D - y
        last_error=D(i)-out2;
        
        %% backpropagation
        
        
        w2 = w2 + (last_error * out1 * eta)';
        delta = last_error * w2 .* (sech(net1).^2)';
        w1 = w1 + eta * (delta' * out0');
        %         for ii=1:size(w1,1)
        %             for j=1:size(w1,2)
        %                 w1(ii,j) = w1(ii,j) + eta * (delta(ii) * out0(j));
        %             end
        %         end
        
    end
    
    mean_square_error=[mean_square_error,mse(error)];
    % validation
    v_error=[];
    
    Yv = [];%network test output
    for i=1:length(pv)
        out0=pv(:,i);
        net1=w1*out0;
        out1=tansig(net1);
        net2=w2*out1;
        out2=purelin(net2);
        v_error=[v_error,Dv(i)-out2];
        Yv = [Yv,out2];
    end
    validation_error(index_of_epochs) = mse(v_error);
    subplot(2,1,1)
    plot(mean_square_error)
    title('MSE error for training data')
    grid on
    subplot(2,1,2)
    plot(validation_error)
    title('MSE error for validation data')
    grid on
    drawnow
    
end
% index_of_epochs
mse_error = mse(error)
validation_error = mse(v_error)
%     pause

% check
t_error=[];
Yt = [];%network check output
for i=1:length(pt)
    out0=pt(:,i);
    net1=w1*out0;
    out1=tansig(net1);
    net2=w2*out1;
    out2=purelin(net2);
    t_error=[t_error,Dt(i)-out2];
    Yt = [Yt,out2];
end
test_error = mse(t_error)

Tr = [Tr mse_error];
Ve = [Ve validation_error];
%     Tc = [Tc check_error];

%% plotting
figure
plot(Y);hold on;plot(D,'r');
title('Outputs for training data & target')
grid on
legend('MLP Output','System Output')

figure
plot(Yv);hold on;plot(Dv,'r');
title('Outputs for validation data & target')
grid on
legend('MLP Output','System Output')

figure
plot(Yt);hold on;plot(Dt,'r');
title('Outputs for test data & target')
grid on
legend('MLP Output','System Output')
