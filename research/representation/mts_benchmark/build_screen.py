"""Metadata-only eligibility screen for the MTS benchmark panel (no pyspi). Writes screen.csv.

Usage: python -m research.representation.mts_benchmark.build_screen RESULTS_DIR OUT_CSV
RESULTS_DIR holds Multiverse `results/multiverse/<clf>/<clf>_balacc.csv` (resample 0) saved as `<clf>_bal.csv`,
from aeon-toolkit/multiverse commit adc21c1 (21 Oct 2026). Metadata: Multiverse paper (arXiv 2603.20352) tables;
channel/class descriptions from that paper and the original archive pages. Verdicts are hand-assigned (MANUAL) or
mechanical (derived channels, M<3, T<100, N<150, saturated >=.97, floor < chance+.05), in that order.
"""
import csv, glob, io, sys
S = sys.argv[1]; OUT = sys.argv[2]
META = """name,M,T,train,test,classes,domain
ArticularyWordRecognition,9,144,275,300,25,Motion
AtrialFibrillation,2,640,15,15,3,Biosignal
BasicMotions,6,100,40,40,4,Motion
CharacterTrajectories,3,60-182,1422,1436,20,Motion
Cricket,6,1197,108,72,12,Motion
DuckDuckGeese,1345,270,50,50,5,Audio
ERing,4,65,30,270,6,Motion
EigenWorms,6,17984,128,131,5,Motion
Epilepsy,3,206,137,138,4,Motion
EthanolConcentration,3,1751,261,263,4,Spectro
FaceDetection,144,62,5890,3524,2,Biosignal
FingerMovements,28,50,316,100,2,Biosignal
HandMovementDirection,10,400,160,74,4,Biosignal
Handwriting,3,152,150,850,26,Motion
Heartbeat,61,405,204,205,2,Audio
InsectWingbeat,200,2-22,25000,25000,10,Audio
JapaneseVowels,12,7-29,270,370,9,Audio
LSST,6,36,2459,2466,14,Sensor
Libras,2,45,180,180,15,Motion
MotorImagery,64,3000,278,100,2,Biosignal
NATOPS,24,51,180,180,6,Motion
PEMS-SF,963,144,267,173,7,Sensor
PenDigits,2,8,7494,3498,10,Motion
PhonemeSpectra,11,217,3315,3353,39,Audio
RacketSports,6,30,151,152,4,Motion
SelfRegulationSCP1,6,896,268,293,2,Biosignal
SelfRegulationSCP2,7,1152,200,180,2,Biosignal
SpokenArabicDigits,13,4-93,6599,2199,10,Audio
StandWalkJump,4,2500,12,15,3,Biosignal
UWaveGestureLibrary,3,315,120,320,8,Motion
Alzheimers,19,15000,45,43,3,Biosignal
Blink,4,510,500,450,2,Biosignal
ButtonPress,32,1000,120,120,2,Biosignal
EyesOpenShut,14,128,56,42,2,Biosignal
FeedbackButton,61,450,1700,1150,2,Biosignal
FeetHands,64,640,2835,1890,2,Biosignal
ImaginedFeetHands,64,640,2835,1935,2,Biosignal
ImaginedOpenCloseFist,64,640,2835,1890,2,Biosignal
InnerSpeech,128,3584,250,200,4,Biosignal
LongIntervalTask,63,4750,1660,1120,2,Biosignal
LowCost,15,375,600,600,2,Biosignal
MatchingPennies,10,10000,700,700,2,Biosignal
MindReading,204,200,727,653,5,Biosignal
OpenCloseFist,64,640,2835,1890,2,Biosignal
PhotoStimulation,19,9000,37,36,3,Biosignal
PronouncedSpeech,128,3584,250,200,4,Biosignal
ShortIntervalTask,63,2750,1360,1420,2,Biosignal
SitStand,17,1250,3200,3200,4,Biosignal
SongFamiliarity,32,3000,700,750,2,Biosignal
VisualSpeech,128,3584,250,200,4,Biosignal
CrowdSourced,14,256,9456,2833,2,Biosignal
DREAMERA,14,256,140638,29608,2,Biosignal
DREAMERV,14,256,140638,29608,2,Biosignal
FordChallenge,30,40,29003,7254,2,Sensor
LenDB,3,540,983483,261459,2,Sensor
Opportunity,113,100,12184,5202,5,Motion
PAMAP2,52,100,28961,9895,12,Motion
S2Agri-17,10,24,47328752,11940071,17,Image
Skoda,60,100,11290,2827,11,Motion
STEW,14,256,21384,7128,2,Biosignal
Tiselac,10,23,79907,19780,9,Image
UCIActivity,9,128,7898,2401,6,Motion
USCActivity,6,100,43060,13168,12,Motion
WISDM,3,100,11989,5177,6,Motion
WISDM2,3,100,98094,50940,6,Motion
IRDS-SFL,75,111,253,116,2,Motion
KERAAL-RTK,77,264,256,14,2,Motion
KIMORE-PR-C,54,847,64,7,2,Motion
KINECAL-QSEO,75,593,68,17,2,Motion
UCDHE-Rowing-MC,16,161,1992,440,5,Motion
UIPRMD-DS-C,66,81,144,36,2,Motion
AppliancesEnergy_disc,24,144,95,42,2,Power
AustraliaRainfall_disc,3,24,112186,48081,4,Sensor
BIDMC32HR_disc,2,4000,5550,2399,3,Biosignal
BeijingPM25Quality_disc,9,24,11918,5048,2,Sensor
HouseholdPowerConsumption1_disc,5,1440,745,686,3,Power
IEEEPPG_disc,5,1000,1768,1328,3,Biosignal
AsphaltRegularityCoordinates,3,66-4201,751,751,2,Motion
AutomotiveRoadTrials,8,300,153,77,2,Sensor
BoneIntensitiesAgeGroup,8,270,600,445,3,Image
CounterMovementJump,3,2001-7069,419,179,3,Motion
EmoPain,30,200,968,355,3,Motion
Locust2022,13,96,6597,3299,2,Sensor
MotionSenseHAR,12,1000,966,265,6,Motion
TactileTextureRecognition,6,2000,1586,681,21,Sensor
"""
CH = dict(  # what the channels are
 ArticularyWordRecognition='EMA sensor coordinates (tongue/lips), 3 sensors x 3 axes', BasicMotions='phone accelerometer+gyroscope axes',
 Cricket='two wrist accelerometers x 3 axes', DuckDuckGeese='audio spectrogram bins', EigenWorms='eigenworm shape coefficients (derived)',
 EthanolConcentration='spectrometer readings (derived)', FaceDetection='MEG sensors', FingerMovements='EEG electrodes', HandMovementDirection='MEG sensors (BCI IV-3)',
 Heartbeat='spectrogram frequency bins (derived)', InsectWingbeat='wingbeat spectra (derived)', JapaneseVowels='LPC cepstral coefficients (derived)',
 LSST='photometric passbands (derived)', MotorImagery='ECoG 8x8 grid, one patient (BCI III-I)', NATOPS='hand/elbow/wrist/thumb coordinates',
 PhonemeSpectra='spectrogram bins (derived)', SelfRegulationSCP1='EEG electrodes, one person (BCI II-Ia)', SelfRegulationSCP2='EEG electrodes, one ALS patient (BCI II-Ib)',
 SpokenArabicDigits='MFCC (derived)', UWaveGestureLibrary='accelerometer axes', Epilepsy='accelerometer axes', Handwriting='accelerometer axes', ERing='finger ring capacitive sensors',
 AtrialFibrillation='ECG leads', StandWalkJump='ECG leads', Libras='hand xy coordinates', PenDigits='pen xy coordinates', CharacterTrajectories='pen xy+force', RacketSports='wrist accelerometer+gyroscope',
 Alzheimers='EEG 10-20 (OpenNeuro ds004504; same source as the in-repo AD/FTD cohort)', PhotoStimulation='EEG 10-20, AD/FTD/control', Blink='EEG (frontal)', ButtonPress='EEG, one participant',
 EyesOpenShut='EEG, one participant', FeedbackButton='EEG', FeetHands='EEG 64 (inferred: PhysioNet EEGMMIDB, Schalk et al.)', ImaginedFeetHands='EEG 64 (inferred: EEGMMIDB)',
 ImaginedOpenCloseFist='EEG 64 (inferred: EEGMMIDB)', OpenCloseFist='EEG 64 (inferred: EEGMMIDB)', InnerSpeech='EEG 128', PronouncedSpeech='EEG 128', VisualSpeech='EEG 128',
 LongIntervalTask='EEG 63 (interval timing, Singh et al. 2023)', ShortIntervalTask='EEG 63 (interval timing, Singh et al. 2023)', LowCost='EEG 15 (low-cost headset)', MatchingPennies='EEG 10',
 MindReading='MEG 204', SitStand='EEG 17', SongFamiliarity='EEG 32', CrowdSourced='EEG 14 (Emotiv-class headsets)', DREAMERA='EEG 14 (Emotiv)', DREAMERV='EEG 14 (Emotiv)', STEW='EEG 14 (Emotiv)',
 FordChallenge='engine sensors (categories only)', LenDB='3-component seismometer', Opportunity='mixed wearable sensors', PAMAP2='mixed IMU + heart rate', S2Agri='satellite spectral bands (derived)',
 Skoda='wearable accelerometers', Tiselac='satellite spectral bands (derived)', UCIActivity='phone IMU (body/total acc, gyro)', USCActivity='IMU', WISDM='accelerometer axes', WISDM2='accelerometer axes',
 EmoPain='26 joint angles + 4 sEMG', AutomotiveRoadTrials='8 on-board diagnostic signals of one vehicle', MotionSenseHAR='phone attitude/gravity/acc/rotation (derived)', TactileTextureRecognition='force/torque components of one sensor',
 Locust2022='13 environmental variables (daily)', CounterMovementJump='accelerometer axes', AsphaltRegularityCoordinates='accelerometer + GPS', BoneIntensitiesAgeGroup='image-derived phalanx profiles',
 UCDHE='2D pose joint coordinates from video', IRDS='skeleton joint coordinates', KERAAL='skeleton joint coordinates', KIMORE='skeleton joint coordinates', KINECAL='skeleton joint coordinates', UIPRMD='skeleton joint coordinates',
)
CLS = dict(MotorImagery='imagined left pinky vs tongue; train/test = different sessions', HandMovementDirection='4 wrist-movement directions', SelfRegulationSCP1='cortical positivity vs negativity (slow cortical potential self-regulation)',
 SelfRegulationSCP2='SCP positivity vs negativity', ImaginedOpenCloseFist='imagined left vs right fist', OpenCloseFist='executed left vs right fist', ImaginedFeetHands='imagined both fists vs both feet', FeetHands='executed both fists vs both feet',
 MatchingPennies='left vs right hand raised', LowCost='idle vs imagined dominant-hand movement', ShortIntervalTask="Parkinson's vs control (3 s interval)", LongIntervalTask="Parkinson's vs control (7 s interval)",
 Alzheimers='AD vs FTD vs control (rest)', PhotoStimulation='AD vs FTD vs control (photic)', CrowdSourced='eyes open vs closed', STEW='high vs low workload', DREAMERA='arousal high/low', DREAMERV='valence high/low',
 AutomotiveRoadTrials='fault vs no fault (300 s segment)', EmoPain='no / low / high pain', TactileTextureRecognition='21 surface textures', MindReading='5 movie categories', FaceDetection='face vs scrambled', BasicMotions='4 activities',
 UCDHE='correct rowing vs error subtypes', Heartbeat='normal vs abnormal heart sound')
CH['PEMS-SF']='freeway occupancy sensors'; CLS['PEMS-SF']='day of week'
MANUAL = {  # verdict, reason
 'ImaginedOpenCloseFist': ('PANEL', 'designed invariance control: left/right is ~ a hemispheric channel relabelling, so z is predicted at chance'),
 'ImaginedFeetHands': ('PANEL', 'paired with ImaginedOpenCloseFist (same montage/subjects); hands vs feet is not a relabelling'),
 'ShortIntervalTask': ('PANEL', 'disease vs control, cross-subject, all shape methods weak (<=.66); subject IDs must be recovered'),
 'AutomotiveRoadTrials': ('PANEL', 'coupled mechanical system; a fault plausibly breaks inter-signal relations; best a-priori z-favoured case'),
 'EmoPain': ('PANEL', 'body coordination under pain (guarding); shape methods unstable; T=200 is below the 250 guide'),
 'SelfRegulationSCP1': ('PANEL', 'positioning against Cliff et al. 2023; predicted univariate-dominated loss (M=6, low-edge)'),
 'HandMovementDirection': ('PANEL', 'predicted topography/waveform-coded loss; cheap; small test set (74)'),
 'BasicMotions': ('POSITIONING', 'Cliff et al. 2023 example; saturated (all ~1.0) and T=100; reported outside the test family'),
 'MotorImagery': ('EXCLUDE', 'floor: all published methods <= .59 BA across the session shift, so no ranking is readable; M64 x T3000 costs ~7.4k SU'),
 'OpenCloseFist': ('OPTIONAL', 'executed replication of the imagined left/right control'),
 'FeetHands': ('OPTIONAL', 'executed replication of the imagined hands/feet contrast'),
 'MatchingPennies': ('EXCLUDE', 'left/right again (redundant with the EEGMMIDB control); T=10000 needs decimation'),
 'LowCost': ('OPTIONAL', 'imagery vs idle; expected ERD (channel-spectral) case'),
 'LongIntervalTask': ('OPTIONAL', 'same cohort as ShortIntervalTask; use only as replication'),
 'Alzheimers': ('EXCLUDE', 'N=88 and same source (ds004504) as the in-repo AD/FTD cohort'),
 'PhotoStimulation': ('EXCLUDE', 'N=73'),
 'CrowdSourced': ('OPTIONAL', 'eyes open/closed: alpha-power positive control for univariate; montage-free transfer candidate'),
 'STEW': ('OPTIONAL', 'workload; T=256; expected theta/alpha spectral (univariate) case'),
 'DREAMERA': ('EXCLUDE', 'self-report affect labels; near-chance cross-subject in the literature (unverified)'),
 'DREAMERV': ('EXCLUDE', 'as DREAMERA'),
 'FeedbackButton': ('EXCLUDE', 'floor: all methods <= .56'), 'SongFamiliarity': ('EXCLUDE', 'floor: all methods <= .54'),
 'MindReading': ('EXCLUDE', 'M=204 and T=200; channel subsetting has no invariance guarantee'),
 'SitStand': ('EXCLUDE', 'labels undocumented in the archive paper'),
 'InnerSpeech': ('EXCLUDE', 'M=128 x T=3584, N=450, best .33 vs chance .25'), 'PronouncedSpeech': ('EXCLUDE', 'as InnerSpeech'), 'VisualSpeech': ('EXCLUDE', 'as InnerSpeech'),
 'ButtonPress': ('EXCLUDE', 'saturated and N=240, one participant'), 'EyesOpenShut': ('EXCLUDE', 'N=98, one participant'),
 'Blink': ('EXCLUDE', 'saturated (catch22 .998); blink amplitude'),
 'SelfRegulationSCP2': ('OPTIONAL', 'floor (all ~.55); M=7; cheap, report outside family if run'),
 'PEMS-SF': ('EXCLUDE', 'saturated (HC2 1.0); M=963 full table is far beyond the sizing table'),
 'TactileTextureRecognition': ('EXCLUDE', 'saturated (>=.997)'), 'Cricket': ('EXCLUDE', 'saturated and N=180'), 'MotionSenseHAR': ('EXCLUDE', 'saturated; derived channels'),
 'LenDB': ('OPTIONAL', '3-component seismogram (polarisation is inter-component dependence), M=3 low-edge; N must be subsampled'),
 'UCDHE-Rowing-MC': ('OPTIONAL', 'coordination errors may be inter-joint lag changes; T=161, 2D coordinates'),
 'FaceDetection': ('EXCLUDE', 'T=62'), 'FingerMovements': ('EXCLUDE', 'T=50'),
 'Opportunity': ('EXCLUDE', 'T=100; HAR is waveform-dominated; mixed sensors'), 'PAMAP2': ('EXCLUDE', 'T=100; mixed sensors'), 'Skoda': ('EXCLUDE', 'T=100; waveform-dominated gestures'),
 'UCIActivity': ('EXCLUDE', 'T=128; waveform-dominated'), 'USCActivity': ('EXCLUDE', 'T=100'),
 'Locust2022': ('EXCLUDE', 'T=96; environmental covariates, not one coupled system'),
 'AsphaltRegularityCoordinates': ('EXCLUDE', 'variable length (66-4201); accelerometer mixed with GPS/velocity'),
}
DERIVED = ('derived',)
UEA = set(META.split(chr(10))[1:31]) and {l.split(',')[0] for l in META.strip().split(chr(10))[1:31]}
res = {}
for f in glob.glob(f'{S}/*_bal.csv'):
    c = f.split('/')[-1][:-8]; r = {}
    for row in csv.reader(open(f)):
        if len(row) >= 2 and row[0] not in ('Resamples:', ''):
            try: r[row[0]] = float(row[1])
            except ValueError: pass
    res[c] = r
rows = []
for m in csv.DictReader(io.StringIO(META)):
    n = m['name']; key = next((k for k in CH if n.startswith(k)), None)
    ch = CH.get(n) or (CH[key] if key else '')
    cls = CLS.get(n) or next((v for k, v in CLS.items() if n.startswith(k)), '')
    best = max(((v[n], c) for c, v in res.items() if n in v), default=None)
    N = int(m['train']) + int(m['test']); K = int(m['classes']); Tm = int(m['T'].split('-')[0]); M = int(m['M'])
    if n in MANUAL and MANUAL[n]:
        verdict, reason = MANUAL[n]
    elif 'derived' in ch or m['domain'] in ('Audio', 'Image', 'Spectro', 'Digital'):
        verdict, reason = 'EXCLUDE', 'channels are derived features, not sensors of one coupled system'
    elif M < 3:
        verdict, reason = 'EXCLUDE', 'M<3: z undefined or degenerate'
    elif Tm < 100:
        verdict, reason = 'EXCLUDE', f'T={m["T"]} too short for the catalogue'
    elif N < 150:
        verdict, reason = 'EXCLUDE', f'N={N}'
    elif best and best[0] >= .97:
        verdict, reason = 'EXCLUDE', 'saturated'
    elif best and best[0] - 1 / K < .05:
        verdict, reason = 'EXCLUDE', 'floor'
    elif 'coordinates' in ch:
        verdict, reason = 'EXCLUDE', 'joint coordinates with small test sets'
    else:
        verdict, reason = 'EXCLUDE', 'no class-coupling rationale beyond waveform/level'
    rows.append(dict(source='UEA30' if n in UEA else ('Multiverse-EEG' if m['domain'] == 'Biosignal' and 'EEG' in ch or 'MEG' in ch else 'Multiverse'),
                     dataset=n, M=M, T=m['T'], n_train=m['train'], n_test=m['test'], classes=K, channels=ch, classes_are=cls,
                     best_published_BA=f'{best[0]:.3f}' if best else '', best_method=best[1] if best else '', chance=f'{1/K:.3f}', verdict=verdict, reason=reason))
EXT = [
 dict(source='external', dataset='ABIDE-PCP (nilearn)', M=39, T='78-316 TRs', n_train='~871 QC (leave-site-out)', n_test='', classes=2, channels='fMRI parcels (MSDL atlas)', classes_are='autism vs control',
      best_published_BA='~.67-.70 acc (Abraham 2017; Dadi 2019, tangent)', best_method='tangent-space connectome', chance='.5', verdict='PANEL', reason='canonical dependence-native benchmark; the SOTA is itself a dependence representation'),
 dict(source='external', dataset='TUAB v3.0.1', M=19, T='one 20 s window at 100 Hz', n_train='2717', n_test='276', classes=2, channels='scalp EEG 10-20, average reference', classes_are='normal vs abnormal clinical EEG',
      best_published_BA='~.85-.86 acc (Gemein 2020)', best_method='handcrafted / deep', chance='.5', verdict='PANEL', reason='large cross-patient clinical benchmark, official split; TUH access exists'),
 dict(source='in-repo banks', dataset='NeuroTycho -> Chennu propofol', M='16 / 19', T='2000@250 Hz / 1000@100 Hz', n_train='480 windows', n_test='20 people', classes=2, channels='macaque bipolar ECoG -> human scalp EEG',
      best_published_BA='', best_method='', chance='.5', verdict='EXTENSION', reason='montage-free transfer; existing r10 banks, harmonised re-cut costs ~100 SU'),
 dict(source='external', dataset='MOABB BNCI2014_001 (BCI IV-2a)', M=22, T='1000 (4 s, 250 Hz)', n_train='9 people x 288', n_test='session 2', classes=4, channels='EEG', classes_are='left/right hand, feet, tongue',
      best_published_BA='', best_method='Riemannian tangent space', chance='.25', verdict='OPTIONAL', reason='duplicates the EEGMMIDB invariance contrast with a session split; add only if EEGMMIDB subject IDs cannot be recovered'),
]

rows += EXT
order = dict(PANEL=0, EXTENSION=1, POSITIONING=2, OPTIONAL=3, EXCLUDE=4)
rows.sort(key=lambda r: (order[r['verdict']], r['source'], r['dataset']))
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter; print(Counter(r['verdict'] for r in rows), len(rows))
