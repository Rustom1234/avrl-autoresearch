import torch,time,math,torch.nn as nn,torch.nn.functional as F
L=512;D=384;NL=6;H=6;V=50304
class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.l1=nn.LayerNorm(D); s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False); s.l2=nn.LayerNorm(D); s.f1=nn.Linear(D,4*D,bias=False); s.f2=nn.Linear(4*D,D,bias=False)
    def forward(s,x):
        B,T,_=x.shape; q,k,v=s.qkv(s.l1(x)).view(B,T,3,H,D//H).permute(2,0,3,1,4)
        x=x+s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D))
        return x+s.f2(F.gelu(s.f1(s.l2(x))))
class G(nn.Module):
    def __init__(s):
        super().__init__(); s.te=nn.Embedding(V,D); s.pe=nn.Embedding(L,D); s.b=nn.ModuleList([Blk() for _ in range(NL)]); s.lf=nn.LayerNorm(D)
    def forward(s,x):
        h=s.te(x)+s.pe(torch.arange(x.shape[1],device=x.device))
        for b in s.b: h=b(h)
        return F.linear(s.lf(h),s.te.weight)
m=G().cuda(); mc=torch.compile(m); print(sum(p.numel() for p in m.parameters())/1e6)
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,fused=True)
B=16
x=torch.randint(0,50257,(B,L+1)).cuda()
t0=time.time()
for i in range(25):
    if i==5: print("compile",time.time()-t0); torch.cuda.synchronize(); t=time.time()
    with torch.autocast("cuda",torch.bfloat16):
        lg=mc(x[:,:-1]); loss=F.cross_entropy(lg.float().view(-1,V),x[:,1:].reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize(); dt=(time.time()-t)/20; print(dt, B*L/dt, torch.cuda.max_memory_allocated()/1e9)
