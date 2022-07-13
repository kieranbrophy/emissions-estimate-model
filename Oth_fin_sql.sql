select distinct fds.ff_v3_ff_basic_af.fsym_id, fds.ff_v3_ff_basic_af.date,
ff_assets, ff_eq_tot, ff_sales, ff_gross_inc, ff_oper_exp_tot, ff_mkt_val
from fds.ff_v3_ff_basic_af
left join fds.ff_v3_ff_basic_cf
on fds.ff_v3_ff_basic_af.fsym_id = fds.ff_v3_ff_basic_cf.fsym_id
where date >= '2020-01-01' 
ORDER BY date desc
