# On the learning mechanisms in physical reasoning

Is dynamics prediction indispensable for physical reasoning? If so, what kind of roles do the dynamics prediction modules play during the physical reasoning process? Most studies focus on designing dynamics prediction networks and treating physical reasoning as a downstream task without investigating the questions above, taking for granted that the designed dynamics prediction would undoubtedly help the reasoning process. In this work, we take a closer look at this assumption, exploring this fundamental hypothesis by comparing two learning mechanisms: LFD and LFI. 

In the **first experiment**, we directly examine and compare these two mechanisms. Results show a surprising finding: Simple LFI is better than or on par with SOTA LFD. 

This observation leads to the **second experiment** with GD, the ideal case of LFD where in dynamics are obtained directly from a simulator. Results show that dynamics, if directly given instead of approximated, would achieve much higher performance than LFI alone on physical reasoning; this essentially serves as the performance upper bound. 

Yet practically,LFD mechanism can only predict AD using dynamics learning modules that mimic the physical laws, making the following downstream physical reasoning modules degenerate into the \ac{lfi} paradigm; see the **third experiment**.We note that this issue is hard to mitigate, as dynamics prediction errors inevitably accumulate in the long horizon. 

Finally, in the **fourth experiment**, we note that LFI, the extremely simpler strategy when done right, is more effective in learning to solve physical reasoning problems. Taken together, the results on the challenging benchmark of PHYRE show that LFI is, if not better, as good as \ac{lfd} with bells and whistles for dynamics prediction. However, the potential improvement from \ac{lfd}, though challenging, remains lucrative. 

![introduction](introduction.pdf)

This is an implementation of learning from intuition (LfI) and learning from dynamics (LfD) on [PHYRE](https://phyre.ai/).

For details, please see [LfI.md](./LfI/LfI.md) and  [LfD.md](./LfD/LfD.md), respectively.
