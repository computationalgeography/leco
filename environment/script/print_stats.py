import pstats
from pstats import SortKey
import sys

stats = pstats.Stats(sys.argv[1])

stats.strip_dirs()

# tottime: time spent in function without time spent in called function
# percall: time spent per call (tottime / ncalls)
stats.sort_stats(SortKey.TIME, SortKey.CALLS)

max_nr_records = 100

stats.print_stats(max_nr_records)
stats.print_callers(max_nr_records)
stats.print_callees(max_nr_records)
