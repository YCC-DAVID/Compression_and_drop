run with modify_model_6.py

the command should like this form``"-epo 160 -fzepo 30 60 -drp 5 10 -tol 1e2 -gma 0.3 -m ssim --cmp_batch_size 8"``, 
which ``epo`` is total epoches, ``fzepo`` is which epoch to freez, ``drp`` represent the drop position, ``tol`` means the compression tolerance,
``gma`` and ``ssim`` are the selection variable, do not need to care, and  ``cmp_batch_size`` represent the compression chunk size
