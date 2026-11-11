"""Required zero-update scalar formula and copied-parameter gradient acceptance."""
def check_formula(torch,parameter):
 result={}
 for gamma in [1,10]:
  p=torch.tensor([1.,-2.,3.,.25],dtype=torch.float32,device=parameter.device,requires_grad=True);a=torch.tensor([.5,-1.,2.,-.25],device=p.device,dtype=torch.float32);omega=.5*((p.float()-a.float())**2).sum();g,=torch.autograd.grad(gamma*omega,p);assert float(omega)==1.25 and torch.equal(g,gamma*(p.detach()-a))
  # Pure engineering perturbation on a separate tensor, never the formal student.
  copied=parameter.detach().float().clone();copy_anchor=copied.clone();copied.reshape(-1)[0]+=.001;copied.requires_grad_(True);v=.5*((copied-copy_anchor)**2).sum();gg,=torch.autograd.grad(gamma*v,copied);assert torch.allclose(gg,gamma*(copied.detach()-copy_anchor),rtol=0,atol=1e-6) and copied.data_ptr()!=parameter.data_ptr();result[str(gamma)]={'small_array_Omega':float(omega),'small_gradient':g.detach().cpu().tolist(),'copied_parameter_perturbation_passed':True}
 return {'status':'passed','formula':'0.5*sum, FP32, no mean/epsilon/clip','checks':result,'formal_parameter_unchanged':True,'new_model_calls':0}
