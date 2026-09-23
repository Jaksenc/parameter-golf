// Native option readout. Protocol: UTF-8 prompt as hex, TAB, uppercase markers.
// Outputs logits, not generated tokens. Every request starts with cleared model memory.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/prctl.h>
#include <signal.h>
#endif
using Clock = std::chrono::steady_clock;
static std::vector<llama_token> tokenize(const llama_vocab* v,const std::string& s,bool special) {
 int32_t n=llama_tokenize(v,s.data(),(int32_t)s.size(),nullptr,0,special,special);
 if(n>=0) throw std::runtime_error("token_count");
 std::vector<llama_token> out(-n);
 n=llama_tokenize(v,s.data(),(int32_t)s.size(),out.data(),out.size(),special,special);
 if(n<0) throw std::runtime_error("tokenize");out.resize(n);return out;
}
static std::string unhex(const std::string& h) {
 if(h.size()%2 || h.size()>4000000) throw std::runtime_error("prompt_size");
 auto val=[](char c)->int { if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("hex");};
 std::string out;out.reserve(h.size()/2);
 for(size_t i=0;i<h.size();i+=2)out.push_back(char(val(h[i])*16+val(h[i+1])));
 return out;
}
int main(int argc,char** argv) {
 if(argc!=3)return 2;
#ifdef __linux__
 prctl(PR_SET_PDEATHSIG,SIGTERM);
#endif
 ggml_backend_load_all_from_path(argv[2]);llama_backend_init();
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;
 auto* model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=8192;cp.n_batch=2048;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.n_outputs_max=1;cp.n_outputs_max_per_seq=1;
 auto* ctx=llama_init_from_model(model,cp);if(!ctx){llama_model_free(model);return 4;}
 const auto* vocab=llama_model_get_vocab(model);
 auto batch=llama_batch_init(2048,0,1);
 std::cout<<"{\"ready\":true,\"n_ctx\":"<<llama_n_ctx(ctx)<<",\"n_batch\":"<<llama_n_batch(ctx)<<",\"n_ubatch\":"<<llama_n_ubatch(ctx)<<",\"vocabulary\":"<<llama_vocab_n_tokens(vocab)<<"}"<<std::endl;
 std::string line;
 while(std::getline(std::cin,line)) {
  try {
   auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("wire");
   std::string prompt=unhex(line.substr(0,tab)),symbols=line.substr(tab+1);
   if(symbols.size()<2||symbols.size()>26)throw std::runtime_error("symbols");
   std::set<char> unique;std::vector<llama_token> ids;
   for(char c:symbols){if(c<'A'||c>'Z'||!unique.insert(c).second)throw std::runtime_error("symbols");auto t=tokenize(vocab,std::string(1,c),false);if(t.size()!=1)throw std::runtime_error("marker_tokens");ids.push_back(t[0]);}
   auto tokens=tokenize(vocab,prompt,true);if(tokens.empty()||tokens.size()>llama_n_ctx(ctx))throw std::runtime_error("context_limit");
   auto start=Clock::now();llama_memory_clear(llama_get_memory(ctx),true);
   llama_set_abort_callback(ctx,[](void* p)->bool{return std::chrono::duration<double>(Clock::now()-*static_cast<Clock::time_point*>(p)).count()>170;},&start);
   for(size_t off=0;off<tokens.size();off+=2048){
    batch.n_tokens=std::min<size_t>(2048,tokens.size()-off);
    for(int i=0;i<batch.n_tokens;i++){batch.token[i]=tokens[off+i];batch.pos[i]=off+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(off+i==tokens.size()-1);}
    if(llama_decode(ctx,batch)!=0)throw std::runtime_error("decode");
   }
   float* z=llama_get_logits_ith(ctx,-1);if(!z)throw std::runtime_error("missing_logits");
   for(auto id:ids)if(!std::isfinite(z[id]))throw std::runtime_error("nonfinite_logit");
   std::cout<<std::setprecision(17)<<"{\"ok\":true,\"prompt_tokens\":"<<tokens.size()<<",\"generated_tokens\":0,\"seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()<<",\"logits\":[";
   for(size_t i=0;i<ids.size();i++){if(i)std::cout<<",";std::cout<<"{\"marker\":\""<<symbols[i]<<"\",\"id\":"<<ids[i]<<",\"logit\":"<<double(z[ids[i]])<<"}";}
   std::cout<<"]}"<<std::endl;
  }catch(const std::exception& e){std::cout<<"{\"ok\":false,\"error\":\""<<e.what()<<"\"}"<<std::endl;}
 }
 llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
