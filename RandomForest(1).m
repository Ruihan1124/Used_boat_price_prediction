load Output
load Input
isCategorical = [zeros(10,1);ones(size(Input,2)-10,1)];
%% Number of Leaves and Trees Optimization
%for RFOptimizationNum=1:5
    
% RFLeaf=[5,10,20,50,100,200,500];
% col='rgbcmyk';
% figure('Name','RF Leaves and Trees');
% for i=1:length(RFLeaf)
%     RFModel=TreeBagger(2000,Input,Output,'Method','R','OOBPrediction','On','CategoricalPredictors',find(isCategorical == 1),'MinLeafSize',RFLeaf(i));
%     plot(oobError(RFModel),col(i));
%     hold on
% end
% xlabel('Number of Grown Trees');
% ylabel('Mean Squared Error') ;
% LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
% title(LeafTreelgd,'Number of Leaves');
% hold off;

%disp(RFOptimizationNum);
%end

%% Notification
% Set breakpoints here.

%% Cycle Preparation
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=500;
for RFCycleRun=1:RFRunNumSet

%% Training Set and Test Set Division
RandomNumber=(randperm(length(Output),floor(length(Output)*0.2)))';
TrainYield=Output;
TestYield=zeros(length(RandomNumber),1);
TrainVARI=Input;
TestVARI=zeros(length(RandomNumber),size(TrainVARI,2));
for i=1:length(RandomNumber)
    m=RandomNumber(i,1);
    TestYield(i,1)=TrainYield(m,1);
    TestVARI(i,:)=TrainVARI(m,:);
    TrainYield(m,1)=0;
    TrainVARI(m,:)=0;
end
TrainYield(all(TrainYield==0,2),:)=[];
TrainVARI(all(TrainVARI==0,2),:)=[];

%% RF
nTree=127;
nLeaf=5;
RFModel=TreeBagger(nTree,TrainVARI,TrainYield,...
    'Method','regression','OOBPredictorImportance','on','CategoricalPredictors',find(isCategorical == 1), 'MinLeafSize',nLeaf);
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel,TestVARI);
% PredictBC107=cellfun(@str2num,PredictBC107(1:end));

%% Accuracy of RF
RFRMSE=sqrt(sum(sum((RFPredictYield-TestYield).^2))/size(TestYield,1));
RFrMatrix=corrcoef(RFPredictYield,TestYield);
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1000
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

%% Variable Importance Contrast
VariableImportanceX={};
XNum=1;
% for TifFileNum=1:length(TifFileNames)
%     if ~(strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeArea') | ...
%             strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeYield'))
%         eval(['VariableImportanceX{1,XNum}=''',TifFileNames(TifFileNum).name(4:end-4),''';']);
%         XNum=XNum+1;
%     end
% end
VariableImportanceX=[{'beam'},{'displacement'},{'draft'},{'feul'},{'Lengthft'},{'sail'},{'Year'},{'GDP'},{'PC'},{'TRV'},{'Make'}]
% for i=1:size(Input,2)
%     eval(['VariableImportanceX{1,XNum}=''',i,''';']);
%     XNum=XNum+1;
% end

figure('Name','Variable Importance Contrast');
VariableImportanceX=categorical(VariableImportanceX);
bar(VariableImportanceX,RFModel.OOBPermutedPredictorDeltaError)
xtickangle(45);
set(gca, 'XDir','normal')
xlabel('Factor');
ylabel('Importance');

yHat = oobPredict(RFModel);
R2 = corr(RFModel.Y,yHat)
r2=corr(RFModel.Y,yHat)^2

Use2=[RFModel.Y,yHat]

figure
plot(Use2)
hold on

%% RF Model Storage
RFModelSavePath='D:\';
save(sprintf('%sRF04.mat',RFModelSavePath),'nLeaf','nTree',...
    'RandomNumber','RFModel','RFPredictConfidenceInterval','RFPredictYield','RFr','RFRMSE',...
    'TestVARI','TestYield','TrainVARI','TrainYield');



