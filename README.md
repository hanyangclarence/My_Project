# Attempt 2: Add full attention to music for motion

## Attempt 2.1: Directly modify the attention mask
In dataloader, 1/3 prob to load full music and motion description, 
1/3 prob to load (mu start) (mu end) (mo start) content (mo end),
and 1/3 ...
Also, I modify the classifier free guidance to drop out the descriptions with  (mu start) (mu end) (mo start) (mo end)
instead of None.