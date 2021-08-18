class RK2(torch.optim.optimizer):
  # ......
  def update_step(self,fwd_bwd_fn):
      """
      fwd_bwd_fn: fn that computes y=model(x); loss(y,yhat).backward()
      example:
        # training loop
        x, yhat=sample_from_dataloader()
        optimizer.zero_grad()

        def fwd_bwd_fn():
          y=model(x)
          loss=compute_loss(y,yhat)
          loss.backward()
          return loss

        fwd_bwd_fn()
        optimizer.update(fwd_bwd_fn)
      """
      #  We want to compute Equation 9 in our paper:
      #  w - n/2* v(w) - n/2 * v(w-n*v(w))
      temp_step = {}  
          with torch.no_grad():
              for jj_, group in enumerate(self.param_groups):
                  #ignore weight-decay for simplicity.

                  for ii_, p in enumerate(group['params']):
                      if p.grad is None:
                          continue
                      d_p = p.grad  # v(w)
                      
                      # first part of the step.
                      # w - n/2*v(w)
                      p1 = p.data.clone().detach()
                      p1.add_(d_p, alpha=-0.5 * group['lr'])
                      temp_step[f"{jj_}_{ii_}"] = p1 #storing for later used.

                      # Computing now w-n*v(w) for the second part of the step
                      p.add_(d_p, alpha=-1.0 * group['lr'])  

                      # set to none
                      p.grad = None  # self.zero_grad()

          # extra step evaluation.
          loss, _ = fwd_bwd_fn()  #  this function do model forward,compute loss and loss backward.
          # gradients are now v(w-n*v(w))

          with torch.no_grad():
              for jj_, group in enumerate(self.param_groups):
                  for ii_, p in enumerate(group['params']):
                      if p.grad is None:
                          continue
                      d_p = p.grad # v(w-n*v(w))

                      # unpacking the first part of the step.
                      p1 = temp_step[f"{jj_}_{ii_}"]

                      # updating p with the stored value from before.
                      # this is equivalent to p = p1 = w-n/2 * v(w) (but sligthly faster)
                      p.zero_()
                      p.add_(p1)

                      #w= w - n/2*v(w) - n/2 * v(w-n*v(w))
                      p.add_(d_p, alpha=-0.5 * group['lr'])

# Source code is based on https://github.com/pytorch/pytorch/blob/release/1.5/torch/optim/sgd.py#L88-L112 and modified to have an extra pass.
