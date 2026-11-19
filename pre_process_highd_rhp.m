function manifest = pre_process_highd_rhp(rawDir, outputDir, varargin)
%PRE_PROCESS_HIGHD_RHP Convert highD CSV recordings to RHP / RiskNet MAT files.
%   PRE_PROCESS_HIGHD_RHP() processes recordings 01--60 from the local highD
%   dataset and writes a highD/ directory under the current working directory.
%   Run from the repository root. Default input: the sibling
%   ../highd-dataset-v1.0/highd-dataset-v1.0/data directory.
%
%   PRE_PROCESS_HIGHD_RHP(RAWDIR, OUTPUTDIR, Name, Value, ...) accepts:
%     'RecordingIds'  subset of 1:60 (default 1:60; useful for a smoke run)
%     'Seed'          vehicle-split random seed (default 42)
%     'SampleStride'  sample-anchor stride in raw 25-Hz frames (default 1)
%     'RiskThreshold' neighbor-selection threshold (default 0.005)
%     'MiniTestCount' maximum number of mini-test samples (default 1000;
%                     set 0 to disable both mini-test files)
%     'Overwrite'     allow replacing this function's outputs (default false)
%
%   Example (MATLAB Command Window, from the RHP repository):
%     rawDir = fullfile('..','highd-dataset-v1.0','highd-dataset-v1.0','data');
%     pre_process_highd_rhp(rawDir, 'highD');
%
%   Outputs and variable names:
%     TrainSetHighD_{15,30,45,60}.mat : trajTr [N x 42]
%     track_HighD_{15,30,45,60}.mat  : tracks [15 x maxVehicleId] cells
%     ValSetHighD.mat, TestSetHighD.mat : traj [N x 42]
%     track_HighD_1.mat, track_HighD_2.mat : tracks [30 x maxVehicleId]
%     test_highd_mini.mat : traj; track_highd_mini.mat : tracks
%     preprocess_manifest.mat : manifest (settings, counts, vehicle splits)
%
%   Each training shard covers 15 recordings in order: 1:15, 16:30, 31:45,
%   46:60. Sample recording IDs remain global; training track-cell row IDs
%   are local to a shard, as required by (dsId-1) mod 15 in loader_highD.py.
%   Concatenating track_HighD_1 and _2 yields 60 globally indexed rows.
%   Empty cells retain missing vehicle IDs; IDs are never renumbered.
%
%   Sample columns (the legacy x/y order is deliberately preserved):
%     [recordingId, vehicleId, frame, y, x, vy, vx, laneId,
%      width, height, ay, ax, 30 neighbor IDs]
%   A nonempty tracks{recordingId, vehicleId} is 10 x T:
%     [frame; y; x; vy; vx; laneId; width; height; ay; ax].
%   Position, velocity, acceleration and dimensions use FEET-based units.
%   Positions are the original CSV reference points, not shifted centers.
%
%   Split policy: shuffle eligible ego vehicles within each recording, then
%   assign floor(70%) / floor(10%) / remainder to train / val / test.
%   All anchors of a (recordingId, vehicleId) stay in ONE target split.
%   Context tracks remain available across splits; this is an ego-vehicle
%   split, not a recording-disjoint or scene-disjoint evaluation protocol.
%   No lane-change oversampling is performed. These files therefore do not
%   reproduce the old random-frame split or its reported scores exactly.
%
%   Based on RiskNet/pre_process_highd.m, with corrected neighbor indexing,
%   current-lane grouping, nonnegative speed scales, bounded memory, and a
%   complete split/save stage. Recording 26 lane-5 rows are excluded as in
%   the reference. A vehicle with nonconsecutive remaining frames is omitted
%   entirely because the Python loader indexes tracks by frame offsets.
%
%   Full test: RiskNet's loader uses TestSetHighD.mat plus track_HighD_1/2.
%   RHP's CURRENT test loader instead uses the mini pair under data/highD/.
%   To use the generated mini pair there, copy BOTH mini files to that
%   location; preserve the existing ego_goal_down/up.mat templates. This
%   function does not overwrite the repository's bundled data or generate
%   intention templates. It does not change any Python loader or config.
%
%   All files use MAT v7, readable by scipy.io.loadmat (not HDF5/v7.3):
%   https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
%   Expect several GB of RAM/disk and a long run for all 60 recordings.

if nargin < 1 || isempty(rawDir)
    rawDir = fullfile('..','highd-dataset-v1.0','highd-dataset-v1.0','data');
end
if nargin < 2 || isempty(outputDir)
    outputDir = 'highD';
end
p = inputParser;
p.addParameter('RecordingIds', 1:60);
p.addParameter('Seed', 42);
p.addParameter('SampleStride', 1);
p.addParameter('RiskThreshold', 0.005);
p.addParameter('MiniTestCount', 1000);
p.addParameter('Overwrite', false);
p.parse(varargin{:});
cfg = p.Results;
validateattributes(cfg.RecordingIds, {'numeric'}, ...
    {'vector','nonempty','integer','>=',1,'<=',60});
cfg.RecordingIds = unique(double(cfg.RecordingIds(:)'));
validateattributes(cfg.Seed, {'numeric'}, {'scalar','integer','>=',0,'<=',2^32-1});
validateattributes(cfg.SampleStride, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(cfg.RiskThreshold, {'numeric'}, {'scalar','nonnegative','finite'});
validateattributes(cfg.MiniTestCount, {'numeric'}, {'scalar','integer','nonnegative','finite'});
validateattributes(cfg.Overwrite, {'logical'}, {'scalar'});
rawDir = char(rawDir);
outputDir = char(outputDir);
assert(isfolder(rawDir), 'RHP:MissingInput', 'Raw data directory not found: %s', rawDir);
stream = RandStream('mt19937ar', 'Seed', cfg.Seed);

% Preflight before creating or overwriting any output.
outputNames = {'ValSetHighD.mat','TestSetHighD.mat', ...
    'track_HighD_1.mat','track_HighD_2.mat','preprocess_manifest.mat'};
for lastId = 15:15:60
    outputNames{end+1} = sprintf('TrainSetHighD_%d.mat', lastId); %#ok<AGROW>
    outputNames{end+1} = sprintf('track_HighD_%d.mat', lastId); %#ok<AGROW>
end
if cfg.MiniTestCount > 0
    outputNames = [outputNames, {'test_highd_mini.mat','track_highd_mini.mat'}];
end
for i = 1:numel(outputNames)
    target = fullfile(outputDir, outputNames{i});
    assert(cfg.Overwrite || ~isfile(target), 'RHP:OutputExists', ...
        'Output exists: %s. Choose a new directory or explicitly set Overwrite=true.', target);
end
metadata = cell(60,1);
maxVehicleId = 0;
for recordingId = cfg.RecordingIds
    prefix = fullfile(rawDir, sprintf('%02d', recordingId));
    for suffix = {'_tracks.csv','_tracksMeta.csv','_recordingMeta.csv'}
        assert(isfile([prefix suffix{1}]), 'RHP:MissingInput', ...
            'Required CSV not found: %s%s', prefix, suffix{1});
    end
    meta = readtable([prefix '_tracksMeta.csv']);
    rec = readtable([prefix '_recordingMeta.csv']);
    requireColumns(meta, {'id','drivingDirection'}, [prefix '_tracksMeta.csv']);
    requireColumns(rec, {'frameRate'}, [prefix '_recordingMeta.csv']);
    assert(height(rec) == 1 && rec.frameRate(1) == 25, 'RHP:FrameRate', ...
        'Recording %d must be sampled at 25 Hz for the existing loaders.', recordingId);
    validateattributes(meta.id, {'numeric'}, {'vector','nonempty','integer','positive','finite'});
    assert(numel(unique(meta.id)) == height(meta), 'RHP:DuplicateId', ...
        'Duplicate vehicle IDs in recording %d metadata.', recordingId);
    assert(all(ismember(meta.drivingDirection, [1 2])), 'RHP:Direction', ...
        'Unexpected drivingDirection in recording %d.', recordingId);
    metadata{recordingId} = meta;
    maxVehicleId = max(maxVehicleId, max(meta.id));
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

valParts = cell(60,1);
testParts = cell(60,1);
splitParts = cell(60,1);
counts = zeros(60,7); % raw rows, retained rows, candidates, train, val, test, dropped vehicles
started = tic;
for shard = 1:4
    recordingIds = cfg.RecordingIds(ceil(cfg.RecordingIds / 15) == shard);
    tracks = cell(15, maxVehicleId);
    trainParts = cell(numel(recordingIds),1);
    for ri = 1:numel(recordingIds)
        recordingId = recordingIds(ri);
        fprintf('\nRecording %02d: reading CSV ...\n', recordingId);
        csvFile = fullfile(rawDir, sprintf('%02d_tracks.csv', recordingId));
        [data, vehicleTracks, anchors, dropped, rawCount] = ...
            readRecording(csvFile, recordingId, maxVehicleId, cfg.SampleStride);
        localId = mod(recordingId - 1, 15) + 1;
        tracks(localId,:) = vehicleTracks;
        meta = metadata{recordingId};
        direction = zeros(maxVehicleId,1);
        direction(meta.id) = meta.drivingDirection;
        assert(all(direction(data(:,2)) > 0), 'RHP:MissingVehicle', ...
            'Recording %d contains a vehicle absent from tracksMeta.', recordingId);
        samples = [data(anchors,:), zeros(numel(anchors),30)];
        fprintf('Recording %02d: %d eligible anchors; computing neighbors ...\n', ...
            recordingId, size(samples,1));
        samples(:,13:42) = buildNeighbors(data, anchors, direction, cfg.RiskThreshold);
        [membership, splitRows] = splitByVehicle(samples(:,2), stream);
        trainParts{ri} = samples(membership == 1,:);
        valParts{recordingId} = samples(membership == 2,:);
        testParts{recordingId} = samples(membership == 3,:);
        splitParts{recordingId} = [repmat(recordingId,size(splitRows,1),1), splitRows];
        counts(recordingId,:) = [rawCount,size(data,1),size(samples,1), ...
            sum(membership==1),sum(membership==2),sum(membership==3),dropped];
        fprintf('Recording %02d: train=%d, val=%d, test=%d (%0.1f min elapsed)\n', ...
            recordingId, counts(recordingId,4:6), toc(started)/60);
        clear data samples vehicleTracks anchors;
    end
    trajTr = concatenateSamples(trainParts);
    clear trainParts;
    lastId = shard * 15;
    writeMat(fullfile(outputDir,sprintf('TrainSetHighD_%d.mat',lastId)), 'trajTr', trajTr);
    clear trajTr;
    writeMat(fullfile(outputDir,sprintf('track_HighD_%d.mat',lastId)), 'tracks', tracks);
    clear tracks;
end

% Validation/test lookups share exactly the same vehicle-ID dimension.
for half = 1:2
    firstShard = 2*half - 1;
    a = load(fullfile(outputDir,sprintf('track_HighD_%d.mat',15*firstShard)), 'tracks');
    b = load(fullfile(outputDir,sprintf('track_HighD_%d.mat',15*(firstShard+1))), 'tracks');
    tracks = [a.tracks; b.tracks];
    clear a b;
    writeMat(fullfile(outputDir,sprintf('track_HighD_%d.mat',half)), 'tracks', tracks);
    clear tracks;
end
traj = concatenateSamples(valParts);
clear valParts;
writeMat(fullfile(outputDir,'ValSetHighD.mat'), 'traj', traj);
clear traj;
traj = concatenateSamples(testParts);
clear testParts;
writeMat(fullfile(outputDir,'TestSetHighD.mat'), 'traj', traj);
miniCount = min(cfg.MiniTestCount, size(traj,1));
if cfg.MiniTestCount > 0
    if miniCount > 0
        mini = traj(round(linspace(1,size(traj,1),miniCount)),:);
    else
        mini = zeros(0,42);
    end
    clear traj;
    writeMat(fullfile(outputDir,'test_highd_mini.mat'), 'traj', mini);
    writeMiniTracks(outputDir, mini, maxVehicleId);
else
    clear traj;
end
manifest = struct;
manifest.rawDirectory = rawDir;
manifest.outputDirectory = outputDir;
manifest.options = cfg;
manifest.splitPolicy = 'Per-recording ego-vehicle split, 70/10/20; seed-controlled; no oversampling';
manifest.splitCodes = {'train','val','test'};
manifest.vehicleSplitColumns = {'recordingId','vehicleId','splitCode','sampleCount'};
manifest.vehicleSplits = vertcat(splitParts{:});
manifest.countColumns = {'rawRows','retainedRows','candidateSamples', ...
    'trainSamples','valSamples','testSamples','droppedDiscontinuousVehicles'};
manifest.countsByRecording = counts;
manifest.miniTestCount = miniCount;
manifest.maxVehicleId = maxVehicleId;
manifest.frameRate = 25;
manifest.historyFrames = 75;
manifest.futureFrames = 125;
manifest.units = 'feet, feet/second, feet/second^2; frame and IDs unchanged';
manifest.neighborSlots = {'leftUp','rightUp','centerUp','leftDown','rightDown','centerDown'};
manifest.neighborSlotSize = 5;
manifest.created = char(datetime('now','Format','yyyyMMdd''T''HHmmss'));
writeMat(fullfile(outputDir,'preprocess_manifest.mat'), 'manifest', manifest);
fprintf('\nDone. Train / val / test samples: %d / %d / %d.\n', sum(counts(:,4:6),1));
fprintf('Outputs: %s\nElapsed: %.1f minutes\n', outputDir, toc(started)/60);
end

function requireColumns(tbl, names, filename)
missing = setdiff(names, tbl.Properties.VariableNames);
assert(isempty(missing), 'RHP:MissingColumn', ...
    'Missing column(s) in %s: %s', filename, strjoin(missing, ', '));
end

function [data, tracks, anchors, dropped, rawCount] = readRecording(filename, recordingId, maxId, stride)
names = {'id','frame','y','x','yVelocity','xVelocity','laneId', ...
    'width','height','yAcceleration','xAcceleration'};
opts = detectImportOptions(filename);
assert(all(ismember(names,opts.VariableNames)), 'RHP:MissingColumn', ...
    'CSV is missing required highD track columns: %s', filename);
opts.SelectedVariableNames = names;
opts = setvartype(opts, names, 'double');
tbl = readtable(filename, opts);
rawCount = height(tbl);
data = [repmat(recordingId,height(tbl),1), tbl{:,names}];
clear tbl;
assert(all(isfinite(data),'all'), 'RHP:InvalidData', 'Nonfinite value in %s.', filename);
validateattributes(data(:,2:3), {'numeric'}, {'integer','positive'});
assert(all(data(:,2) <= maxId), 'RHP:MissingVehicle', 'Vehicle ID exceeds metadata in %s.', filename);
assert(all(data(:,9:10) > 0,'all'), 'RHP:Dimensions', 'Invalid vehicle dimension in %s.', filename);
if recordingId == 26
    data(data(:,8) == 5,:) = [];
end
% Convert ONLY kinematic/dimension columns; frame, lane, IDs stay integers.
data(:,[4:7,9:12]) = data(:,[4:7,9:12]) / 0.3048;
data = sortrows(data,[2 3]);
tracks = cell(1,maxId);
keep = true(size(data,1),1);
eligible = false(size(data,1),1);
dropped = 0;
if isempty(data)
    anchors = zeros(0,1);
    return;
end
starts = [1; find(diff(data(:,2)) ~= 0)+1];
ends = [starts(2:end)-1; size(data,1)];
for i = 1:numel(starts)
    rows = starts(i):ends(i);
    vehicleId = data(starts(i),2);
    if any(diff(data(rows,3)) ~= 1)
        warning('RHP:DiscontinuousTrack', ...
            'Omitting recording %d vehicle %d: frames are not unique and consecutive.', recordingId,vehicleId);
        keep(rows) = false;
        dropped = dropped + 1;
        continue;
    end
    tracks{vehicleId} = data(rows,3:12)';
    eligible(starts(i)+75:stride:ends(i)-125) = true;
end
data = data(keep,:);
anchors = find(eligible(keep));
end

function neighbors = buildNeighbors(data, anchors, direction, threshold)
neighbors = zeros(numel(anchors),30);
if isempty(anchors)
    return;
end
sampleIndex = zeros(size(data,1),1);
sampleIndex(anchors) = (1:numel(anchors))';
[~,order] = sortrows(data(:,[3 2]),[1 2]);
frames = data(order,3);
starts = [1; find(diff(frames) ~= 0)+1];
ends = [starts(2:end)-1; numel(order)];
for fi = 1:numel(starts)
    rows = order(starts(fi):ends(fi));
    ego = find(sampleIndex(rows) > 0);
    if isempty(ego)
        continue;
    end
    a = data(rows,:);
    % Pairwise geometry in meters, using the original highD coordinate axes.
    x = a(:,5)*0.3048; y = a(:,4)*0.3048;
    vx = a(:,7)*0.3048; vy = a(:,6)*0.3048;
    w = a(:,9)*0.3048; h = a(:,10)*0.3048;
    dx0 = x(ego)-x'; dy0 = y(ego)-y';
    dx = max(abs(dx0)-0.5*(w(ego)+w'),0).*dx0./(abs(dx0)+1e-2);
    dy = max(abs(dy0)-0.5*(h(ego)+h'),0).*dy0./(abs(dy0)+1e-2);
    dvx = -(vx(ego)-vx').*dx./(abs(dx)+1e-2);
    dvy = -(vy(ego)-vy').*dy./(abs(dy)+1e-2);
    % Keep the reference selection field; abs(speed) handles both directions.
    speed = abs(vx(ego));
    gx = speed*1.5+2; gy = speed*0.03+1; bx = speed*0.04+2;
    subjective = exp(-abs(dx./gx).^bx-abs(dy./gy).^5.5);
    v = max(hypot(dvx,dvy),1e-4);
    dm2 = (dvx.*dy-dvy.*dx).^2./v.^2;
    td = -(dvx.*dx+dvy.*dy)./v.^2;
    td = 1./max(1./td,1e-4);
    objective = exp(-(dm2/3).^5).*exp(-(td/5).^4);
    allowed = direction(a(ego,2)) == direction(a(:,2))';
    allowed = allowed & (a(ego,2) ~= a(:,2)');
    allowed = allowed & (subjective > threshold | objective > threshold);
    % Up/down mean larger/smaller raw x, NOT direction-normalized front/rear.
    up = x' > x(ego);
    left = a(:,8)' < a(ego,8);
    right = a(:,8)' > a(ego,8);
    center = a(:,8)' == a(ego,8);
    buckets = {left & up, right & up, center & up, ...
        left & ~up, right & ~up, center & ~up};
    distance = hypot(dx0,dy0);
    out = zeros(numel(ego),30);
    k = min(5,size(a,1));
    for bucket = 1:6
        d = distance;
        d(~(allowed & buckets{bucket})) = Inf;
        [best, idx] = sort(d,2,'ascend');
        idx = idx(:,1:k);
        ids = reshape(a(idx,2),size(idx));
        ids(~isfinite(best(:,1:k))) = 0;
        out(:,(bucket-1)*5+(1:k)) = ids;
    end
    neighbors(sampleIndex(rows(ego)),:) = out;
    if mod(fi,5000) == 0
        fprintf('  frames %d / %d\n', fi, numel(starts));
    end
end
end

function [membership, splitRows] = splitByVehicle(egoIds, stream)
[ids,~,group] = unique(egoIds);
n = numel(ids);
permutation = randperm(stream,n);
labels = 3*ones(n,1);
nTrain = floor(0.7*n);
nVal = floor(0.1*n);
labels(permutation(1:nTrain)) = 1;
labels(permutation(nTrain+1:nTrain+nVal)) = 2;
membership = labels(group);
if n == 0
    splitRows = zeros(0,3);
else
    splitRows = [ids, labels, accumarray(group,1,[n 1])];
end
end

function result = concatenateSamples(parts)
parts = parts(~cellfun(@isempty,parts));
if isempty(parts)
    result = zeros(0,42);
else
    result = vertcat(parts{:});
end
end

function writeMiniTracks(outputDir, mini, maxId)
% Preserve global recording/vehicle IDs while storing only referenced tracks.
tracks = cell(60,maxId);
for shard = 1:4
    recordings = unique(mini(ceil(mini(:,1)/15)==shard,1))';
    if isempty(recordings)
        continue;
    end
    s = load(fullfile(outputDir,sprintf('track_HighD_%d.mat',shard*15)), 'tracks');
    for recordingId = recordings
        a = mini(mini(:,1)==recordingId,:);
        ids = unique([a(:,2); reshape(a(:,13:42),[],1)]);
        ids(ids==0) = [];
        localId = mod(recordingId-1,15)+1;
        assert(all(~cellfun(@isempty,s.tracks(localId,ids))), ...
            'RHP:MissingTrack','Mini sample refers to a missing track in recording %d.',recordingId);
        tracks(recordingId,ids) = s.tracks(localId,ids);
    end
    clear s;
end
writeMat(fullfile(outputDir,'track_highd_mini.mat'), 'tracks', tracks);
end

function writeMat(filename, variableName, value)
% Fail explicitly instead of silently emitting v7.3 files the loader cannot read.
info = whos('value');
assert(info.bytes < 2^31-2^20, 'RHP:MatV7Limit', ...
    ['Variable %s exceeds the MAT v7 size limit in %s. Increase SampleStride ' ...
     'or use smaller shards with corresponding loader changes.'], variableName,filename);
payload = struct;
payload.(variableName) = value;
save(filename,'-struct','payload','-v7');
saved = whos('-file',filename);
assert(isscalar(saved) && strcmp(saved.name,variableName) && isequal(saved.size,size(value)), ...
    'RHP:SaveFailed','Output verification failed: %s',filename);
fprintf('Saved %s [%s]\n', filename, num2str(size(value)));
end
