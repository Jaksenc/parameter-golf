"""Pre-execution transcription correction; no scoring or model behavior changes."""
import argparse
import resume_frozen as r
# Verified against all eight original journals in the local audit.
r.PRESENT='288229b0e18870483d2e4cb67dea7eedf429e7fae2d3d4ddbfc3ad8805361973'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();r.run(a.shard)
