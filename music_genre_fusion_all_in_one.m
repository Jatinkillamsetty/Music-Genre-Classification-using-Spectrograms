% music_genre_fusion_all_in_one.m
rng(0);
dataRoot = uigetdir(pwd, 'Select dataset root (4 genre subfolders)');
fsTarget = 22050;
trackDurationSec = 30;
gmmComponents = 8;
trainRatio = 0.7;
valRatioWithinTrain = 0.2;
maxFramesPerGenreForGMM = 20000;

d = dir(dataRoot); d = d([d.isdir]); d = d(~startsWith({d.name}, '.'));
if numel(d) < 4, error('Need at least 4 genre subfolders'); end
d = d(1:4); genreNames = string({d.name});

files = {}; labels = [];
for g = 1:4
    aud = dir(fullfile(dataRoot, d(g).name, '*.wav'));
    for k = 1:numel(aud)
        files{end+1,1} = fullfile(aud(k).folder, aud(k).name);
        labels(end+1,1) = g;
    end
end
files = files(:); labels = labels(:); nFiles = numel(files);

trackFeatCell = cell(nFiles,1);
frameMFCCs = cell(nFiles,1);
for i = 1:nFiles
    [x,fs] = audioread(files{i});
    if size(x,2)>1, x = mean(x,2); end
    if fs~=fsTarget, x = resample(x,fsTarget,fs); fs = fsTarget; end
    N = trackDurationSec*fs;
    if numel(x) < N, x(end+1:N,1) = 0; else x = x(1:N); end
    x = x / (max(abs(x)) + eps);
    mf = mfcc(x, fs, 'NumCoeffs',13);
    dmf = [zeros(1,13); diff(mf)];
    sc = spectralCentroid(x, fs);
    sb = spectralBandwidth(x, fs);
    sr = spectralRolloffPoint(x, fs);
    zc = zerocrossrate(x, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    r = rms(x, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    sf = spectralFlatness(x, fs, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    feat = [mean(mf,1), std(mf,0,1), mean(dmf,1), std(dmf,0,1), ...
            mean(sc), std(sc), mean(sb), std(sb), mean(sr), std(sr), ...
            mean(zc), std(zc), mean(r), std(r), mean(sf), std(sf)];
    trackFeatCell{i} = feat;
    frameMFCCs{i} = mf;
end
X = cell2mat(trackFeatCell); Y = labels;

trainIdx = false(nFiles,1); testIdx = false(nFiles,1);
for g = 1:4
    idx = find(Y==g); k = numel(idx); rp = randperm(k); t = max(1,floor(trainRatio*k));
    trainIdx(idx(rp(1:t))) = true; testIdx(idx(rp(t+1:end))) = true;
end

trainList = find(trainIdx);
nTrain = numel(trainList);
nVal = max(1, round(valRatioWithinTrain * nTrain));
rp = randperm(nTrain);
valList = trainList(rp(1:nVal));
train2List = trainList(rp(nVal+1:end));
train2Idx = false(nFiles,1); valIdx = false(nFiles,1);
train2Idx(train2List) = true; valIdx(valList) = true;

framesPerGenre = cell(4,1);
for g = 1:4
    idx = find(train2Idx & (Y==g));
    frames = [];
    for ii = 1:numel(idx)
        mf = frameMFCCs{idx(ii)};
        frames = [frames; mf];
        if size(frames,1) >= maxFramesPerGenreForGMM
            frames = frames(1:maxFramesPerGenreForGMM,:);
            break;
        end
    end
    framesPerGenre{g} = frames;
end

gmms = cell(4,1);
for g = 1:4
    opts = statset('MaxIter',500);
    try
        gmms{g} = fitgmdist(framesPerGenre{g}, gmmComponents, 'RegularizationValue',1e-6, 'Options',opts);
    catch
        gmms{g} = fitgmdist(framesPerGenre{g}, max(1,min(4,floor(size(framesPerGenre{g},1)/10))), 'RegularizationValue',1e-6, 'Options',opts);
    end
end

svmModel = fitcecoc(X(train2Idx,:), Y(train2Idx), 'Coding','onevsall');

valListIdx = find(valIdx);
nValActual = numel(valListIdx);
gmmProbsVal = zeros(nValActual,4);
svmProbsVal = zeros(nValActual,4);
for t = 1:nValActual
    i = valListIdx(t);
    mf = frameMFCCs{i};
    ll = zeros(4,1);
    for g = 1:4
        p = pdf(gmms{g}, mf);
        ll(g) = sum(log(p + realmin));
    end
    s = ll - max(ll); p_gmm = exp(s); p_gmm = p_gmm / sum(p_gmm);
    gmmProbsVal(t,:) = p_gmm';
    [~,scores] = predict(svmModel, X(i,:));
    s2 = scores - max(scores); p_svm = exp(s2); p_svm = p_svm / sum(p_svm);
    svmProbsVal(t,:) = p_svm;
end

alphas = 0:0.05:1;
bestA = 0.5; bestAcc = -inf;
for a = alphas
    fused = a*gmmProbsVal + (1-a)*svmProbsVal;
    [~,pred] = max(fused,[],2);
    acc = mean(pred == Y(valIdx));
    if acc > bestAcc, bestAcc = acc; bestA = a; end
end
fprintf('Selected fusion weight alpha = %.2f (val acc = %.2f%%)\n', bestA, bestAcc*100);

framesPerGenre_full = cell(4,1);
for g = 1:4
    idx = find(trainIdx & (Y==g));
    frames = [];
    for ii = 1:numel(idx)
        mf = frameMFCCs{idx(ii)};
        frames = [frames; mf];
        if size(frames,1) >= maxFramesPerGenreForGMM
            frames = frames(1:maxFramesPerGenreForGMM,:);
            break;
        end
    end
    framesPerGenre_full{g} = frames;
end
gmms_full = cell(4,1);
for g = 1:4
    opts = statset('MaxIter',500);
    try
        gmms_full{g} = fitgmdist(framesPerGenre_full{g}, gmmComponents, 'RegularizationValue',1e-6, 'Options',opts);
    catch
        gmms_full{g} = fitgmdist(framesPerGenre_full{g}, max(1,min(4,floor(size(framesPerGenre_full{g},1)/10))), 'RegularizationValue',1e-6, 'Options',opts);
    end
end
svmModel_full = fitcecoc(X(trainIdx,:), Y(trainIdx), 'Coding','onevsall');

testList = find(testIdx); nTest = numel(testList);
predGMM = zeros(nTest,1); predSVM = zeros(nTest,1);
predMaj = zeros(nTest,1); predFused = zeros(nTest,1);
gmmProbsTest = zeros(nTest,4); svmProbsTest = zeros(nTest,4);
for t = 1:nTest
    i = testList(t);
    mf = frameMFCCs{i};
    ll = zeros(4,1);
    for g = 1:4
        p = pdf(gmms_full{g}, mf);
        ll(g) = sum(log(p + realmin));
    end
    s = ll - max(ll); p_gmm = exp(s); p_gmm = p_gmm / sum(p_gmm);
    gmmProbsTest(t,:) = p_gmm';
    [lbl, scores] = predict(svmModel_full, X(i,:));
    s2 = scores - max(scores); p_svm = exp(s2); p_svm = p_svm / sum(p_svm);
    svmProbsTest(t,:) = p_svm;
    [~,predGMM(t)] = max(p_gmm);
    predSVM(t) = lbl;
    if predGMM(t) == predSVM(t)
        predMaj(t) = predSVM(t);
    else
        predMaj(t) = predSVM(t);
    end
    fused = bestA*p_gmm' + (1-bestA)*p_svm;
    [~,predFused(t)] = max(fused);
end
trueTest = Y(testIdx);
accGMM = mean(predGMM==trueTest);
accSVM = mean(predSVM==trueTest);
accMaj = mean(predMaj==trueTest);
accFused = mean(predFused==trueTest);
fprintf('Test Accuracies: GMM = %.2f%% | SVM = %.2f%% | Majority = %.2f%% | Fused(alpha=%.2f) = %.2f%%\n', ...
    accGMM*100, accSVM*100, accMaj*100, bestA, accFused*100);

figure('Name','GMM Confusion'); confusionchart(trueTest, predGMM, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
figure('Name','SVM Confusion'); confusionchart(trueTest, predSVM, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
figure('Name','Fused Confusion'); confusionchart(trueTest, predFused, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

save('genre_models_fusion.mat','svmModel_full','gmms_full','genreNames','bestA');

[file, path] = uigetfile('*.wav','Select audio file to classify (optional)');
if ischar(file)
    [x, fs] = audioread(fullfile(path,file));
    if size(x,2)>1, x = mean(x,2); end
    if fs~=fsTarget, x = resample(x, fsTarget, fs); fs = fsTarget; end
    N = trackDurationSec*fs;
    if numel(x) < N, x(end+1:N,1) = 0; else x = x(1:N); end
    x = x / (max(abs(x)) + eps);
    mf = mfcc(x, fs, 'NumCoeffs',13);
    dmf = [zeros(1,13); diff(mf)];
    sc = spectralCentroid(x, fs);
    sb = spectralBandwidth(x, fs);
    sr = spectralRolloffPoint(x, fs);
    zc = zerocrossrate(x, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    r = rms(x, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    sf = spectralFlatness(x, fs, 'WindowLength',round(0.046*fs),'OverlapLength',round(0.023*fs));
    feat = [mean(mf,1), std(mf,0,1), mean(dmf,1), std(dmf,0,1), ...
            mean(sc), std(sc), mean(sb), std(sb), mean(sr), std(sr), ...
            mean(zc), std(zc), mean(r), std(r), mean(sf), std(sf)];
    ll = zeros(4,1);
    for g = 1:4
        p = pdf(gmms_full{g}, mf);
        ll(g) = sum(log(p + realmin));
    end
    s = ll - max(ll); p_gmm = exp(s); p_gmm = p_gmm / sum(p_gmm);
    [lbl, scores] = predict(svmModel_full, feat);
    s2 = scores - max(scores); p_svm = exp(s2); p_svm = p_svm / sum(p_svm);
    [~,gmmPred] = max(p_gmm);
    svmPred = lbl;
    if gmmPred == svmPred, majPred = svmPred; else majPred = svmPred; end
    fusedP = bestA*p_gmm' + (1-bestA)*p_svm;
    [~,fusedPred] = max(fusedP);
    fprintf('File: %s\n', file);
    fprintf('GMM predicted genre: %s\n', genreNames(gmmPred));
    fprintf('SVM predicted genre: %s\n', genreNames(svmPred));
    fprintf('Majority predicted genre: %s\n', genreNames(majPred));
    fprintf('Fused predicted genre (alpha=%.2f): %s\n', bestA, genreNames(fusedPred));
end
