"""Cross-Lingual Safety Audit — analysis & figures (4 models). Operates only on labels."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import fisher_exact
import os
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
    "axes.titleweight":"bold","axes.labelsize":11,"axes.edgecolor":"#444","axes.linewidth":0.8,"figure.dpi":300})

LANG = ["English","Zulu","Venda","Sepedi"]
MODEL = ["Sonnet 4.6","Gemma 3:4b","phi-4","Qwen 2.5:72b"]
CAT = ["Ethnic incitement","Political disinformation","Grant/financial scams","Gender-based violence"]
SLAB = {0:"Refusal",1:"Partial compliance",2:"Complete compliance",3:"Cross-lingual alignment failure"}
SCOL = {0:"#2e7d32",1:"#f9a825",2:"#c62828",3:"#5c6bc0"}

df = pd.read_csv("data/results_clean.csv")
def rate(s,c): return 100.0*(s["response_status"]==c).mean()

g = df.groupby(["model","language"],observed=True)
summary = g.apply(lambda s: pd.Series({"n":len(s),"refusal_%":rate(s,0),"partial_%":rate(s,1),
    "complete_%":rate(s,2),"alignfail_%":rate(s,3),"any_compliance_%":rate(s,1)+rate(s,2)}),
    include_groups=False).round(1).reset_index()
refusal = summary.pivot(index="model",columns="language",values="refusal_%").reindex(MODEL,axis=0).reindex(LANG,axis=1)
comply  = summary.pivot(index="model",columns="language",values="any_compliance_%").reindex(MODEL,axis=0).reindex(LANG,axis=1)
afail   = summary.pivot(index="model",columns="language",values="alignfail_%").reindex(MODEL,axis=0).reindex(LANG,axis=1)
catref  = df.groupby(["model","category"],observed=True).apply(lambda s:rate(s,0),include_groups=False).unstack("category").reindex(MODEL,axis=0).reindex(CAT,axis=1)

print("### REFUSAL RATE (%)\n", refusal.to_string())
print("\n### ANY-COMPLIANCE (%)\n", comply.to_string())
print("\n### ALIGNMENT-FAILURE (%)\n", afail.to_string())
print("\n### REFUSAL by category (%)\n", catref.round(1).to_string())
summary.to_csv("results/summary_by_model_language.csv",index=False)
refusal.to_csv("results/refusal_rate_pivot.csv")

# significance
print("\n### SIGNIFICANCE (Fisher exact, refusal English vs pooled indigenous)")
dd = df.copy(); dd["ref"]=(dd.response_status==0).astype(int)
for m in MODEL:
    s=dd[dd.model==m]; e=s[s.language=="English"]["ref"]; i=s[s.language!="English"]["ref"]
    a,b=int(e.sum()),int(len(e)-e.sum()); c,d=int(i.sum()),int(len(i)-i.sum())
    _,p=fisher_exact([[a,b],[c,d]]); print(f"  {m:14s} Eng {a}/12 vs Indig {c}/36  p={p:.4f}")
op=dd[dd.model!="Sonnet 4.6"]; e=op[op.language=="English"]["ref"]; i=op[op.language!="English"]["ref"]
a,b=int(e.sum()),int(len(e)-e.sum()); c,d=int(i.sum()),int(len(i)-i.sum()); _,p=fisher_exact([[a,b],[c,d]])
print(f"  POOLED open (Gemma+phi4+Qwen) Eng {a}/36 ({100*a/36:.0f}%) vs Indig {c}/108 ({100*c/108:.0f}%)  p={p:.2e}")

# ---- FIG 1: composition, 2x2 ----
fig,axes = plt.subplots(2,2,figsize=(11,7.2),sharey=True)
for ax,model in zip(axes.ravel(),MODEL):
    sub=df[df.model==model]
    comp=pd.DataFrame({c:[100*((sub[sub.language==L].response_status==c).mean()) for L in LANG] for c in [0,1,2,3]},index=LANG)
    bottom=np.zeros(len(LANG))
    for c in [0,1,2,3]:
        ax.bar(LANG,comp[c],bottom=bottom,color=SCOL[c],width=0.66,edgecolor="white",linewidth=0.6)
        for i,(v,bb) in enumerate(zip(comp[c],bottom)):
            if v>=10: ax.text(i,bb+v/2,f"{v:.0f}",ha="center",va="center",color="white",fontsize=8,fontweight="bold")
        bottom+=comp[c].values
    ax.set_title(model); ax.set_ylim(0,100); ax.margins(x=0.04)
    for s in ("top","right"): ax.spines[s].set_visible(False)
for ax in axes[:,0]: ax.set_ylabel("Share of responses (%)")
handles=[Patch(facecolor=SCOL[c],label=SLAB[c]) for c in [0,1,2,3]]
fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.02),fontsize=9)
fig.suptitle("Response composition by language and model  (n = 12 prompts per cell)",fontsize=13,fontweight="bold",y=1.0)
fig.tight_layout(rect=[0,0.03,1,0.99]); fig.savefig("figures/fig1_response_composition.pdf",bbox_inches="tight"); fig.savefig("figures/fig1_response_composition.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---- FIG 2: heatmap ----
fig,ax=plt.subplots(figsize=(7.2,4.0)); M=refusal.values.astype(float)
im=ax.imshow(M,cmap="RdYlGn",vmin=0,vmax=100,aspect="auto")
ax.set_xticks(range(len(LANG)),LANG); ax.set_yticks(range(len(MODEL)),MODEL)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]; ax.text(j,i,f"{v:.0f}%",ha="center",va="center",color="#111" if 25<=v<=80 else "white",fontweight="bold",fontsize=11)
ax.set_title("Refusal rate by model and language"); ax.set_xlabel("Language"); ax.set_ylabel("Model")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label("Refusal rate (%)",fontsize=9)
ax.set_xticks(np.arange(-.5,len(LANG),1),minor=True); ax.set_yticks(np.arange(-.5,len(MODEL),1),minor=True)
ax.grid(which="minor",color="white",linewidth=2); ax.tick_params(which="minor",length=0)
fig.tight_layout(); fig.savefig("figures/fig2_refusal_heatmap.pdf",bbox_inches="tight"); fig.savefig("figures/fig2_refusal_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# ---- FIG 3: category bars ----
fig,ax=plt.subplots(figsize=(11,4.8)); x=np.arange(len(CAT)); w=0.2
mc={"Sonnet 4.6":"#1565c0","Gemma 3:4b":"#6a1b9a","phi-4":"#00838f","Qwen 2.5:72b":"#ef6c00"}
for k,model in enumerate(MODEL):
    vals=catref.loc[model].values
    bars=ax.bar(x+(k-1.5)*w,vals,width=w,label=model,color=mc[model],edgecolor="white",linewidth=0.5)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1.5,f"{v:.0f}",ha="center",va="bottom",fontsize=7.5,fontweight="bold",color="#333")
ax.set_xticks(x,[c.replace(" ","\n",1) for c in CAT],fontsize=9.5); ax.set_ylabel("Refusal rate (%)"); ax.set_ylim(0,112)
ax.set_title("Refusal rate by harm category (averaged across all four languages)")
ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.12))
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",linestyle=":",alpha=0.5)
fig.tight_layout(); fig.savefig("figures/fig3_category_refusal.pdf",bbox_inches="tight"); fig.savefig("figures/fig3_category_refusal.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("\nFigures regenerated for 4 models.")
