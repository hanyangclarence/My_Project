# Attempt 2: Add full attention to music for motion

## Attempt 2.1: Directly modify the attention mask
In dataloader, 1/3 prob to load full music and motion description, 
1/3 prob to load (mu start) (mu end) (mo start) content (mo end),
and 1/3 ...
Also, I modify the classifier free guidance to drop out the descriptions with  (mu start) (mu end) (mo start) (mo end)
instead of None.

    motion: 5s generation
    e18 gs4: 'fid_k': 529.6752637892415, 'fid_m': (37.91976025333658-4.90817662574096e-09j), 'div_k': 14.847358729126436, 'div_m': 7.175902336975979 (but this one look really bad)
    e24 gs4: 'fid_k': 78.75371202474025, 'fid_m': (27.668118089107793-1.3771526696731272e-08j), 'div_k': 10.203726012317619, 'div_m': 6.3378763021992475  (this actually looks pretty good!)
    e30 gs4: 'fid_k': 19.648113510289193, 'fid_m': (110.73815681604201-1.636044377507591e-08j), 'div_k': 6.890160970258482, 'div_m': 8.028417848993472  (this look bad again. there is a lot of jittering)

    motion: 20s generation on aist
    e18 gs4: 'fid_k': (1005.4971291774298-9.347177361498772e-06j), 'fid_m': 47.067750589072105, 'div_k': 25.820936558261895, 'div_m': 5.537378755746744
    e24 gs4: 'fid_k': (17320.742666420585-2.5684689917775003e-05j), 'fid_m': 28.34650509996235, 'div_k': 83.08212972298647, 'div_m': 4.0527641731959125, 'div_k_gt': 9.250034450753242, 'div_m_gt': 7.342270033142343  (so strange..)
    e30 gs4: 'fid_k': (4386.197219640214-1.4021361884247285e-05j), 'fid_m': (85.65302238392633-8.685384964256249e-09j), 'div_k': 54.52077571734404, 'div_m': 5.429701764614154

!! Then maybe when generating caption, we can also do this. So that we can generate 
music and motion caption separately.

## Attempt 2.2: Maybe add another task embedding for different generation task
Since music can sometime attend to motion (previous or all) and sometimes not, it may confuse it.
So just add another embedding to signify which task it is in. 
