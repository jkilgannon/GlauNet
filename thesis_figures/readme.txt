Correctness: Number of pixels that are identified correctly
Incorrectness: Number of pixels that are identified correctly
Aggregate correctness: correct pixels minus incorrect pixels


Three different batches of correct/incorrect/aggregate graphs:

1) Shows up to 70 epochs, and stops there, but shows all of the LRs looked at. Had to 
stop there b/c LR 10e-2 stopped early (due to not training) at Epoch 73.  

2) Removes LR 10e-2, 10e-3, and 10e-4 b/c they were all clearly not training 
(and all stopped early b/c of that), and instead focuses on the other LRs.  This 
second set of graphs stops at Epoch 250 b/c LR 10e-6 stopped early (due to not 
training) at Epoch 253.  This is sufficient to show that 10e-6 and 10e-7 are out 
of contention.

3) Has only 10e-5 and 5 * 10e-6, because these are the only LRs still training.  It 
runs to 500 epochs.  Shows that 10e-5 is superior.
