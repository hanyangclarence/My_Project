# Attempt 2: Add full attention to music for motion

## Attempt 2.1: Directly modify the attention mask
In dataloader, 1/3 prob to load full music and motion description, 
1/3 prob to load (mu start) (mu end) (mo start) content (mo end),
and 1/3 ...
Also, I modify the classifier free guidance to drop out the descriptions with  (mu start) (mu end) (mo start) (mo end)
instead of None.

!! Then maybe when generating caption, we can also do this. So that we can generate 
music and motion caption separately.

## Attempt 2.2: Maybe add another task embedding for different generation task
Since music can sometime attend to motion (previous or all) and sometimes not, it may confuse it.
So just add another embedding to signify which task it is in. 
