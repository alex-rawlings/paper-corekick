functions {
    vector relation_lin(vector x, real a){
        return a .* x + 1.;
    }

    real relation_lin(real x, real a){
        return a * x + 1.;
    }

    real partial_sum(array[] real y_slice, int start, int end, vector vk, real a, real err){
        return normal_lpdf( y_slice | relation_lin(vk[start:end], a), err);
    }

    #include "../hierarchical_models/stan/custom_rngs.stan"
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of observed velocity values in units of v_esc
    vector<lower=0>[N_tot] vkick;
    // array of observed core radii in units of rb0
    vector<lower=0>[N_tot] rb;

    // Out of Sample points
    // follows same structure as above
    // total number of OOS points
    int<lower=1> N_OOS;
    // OOS kick velocities
    vector<lower=0>[N_OOS] vkick_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
    vector<lower=0>[N_GQ] vkick_GQ = append_row(vkick, vkick_OOS);
}


parameters {
    real<lower=0.> a;
    real<lower=0.> err;
}


transformed parameters {
    array[2] real lprior;
    lprior[1] = normal_lpdf(a | 5, 3.);
    lprior[2] = normal_lpdf(err | 0, 9);
}


model {
    // density at priors
    target += sum(lprior);

    target += reduce_sum(partial_sum, to_array_1d(rb), 1, vkick, a, err);

}


generated quantities {
    // generate data replication
    vector[N_GQ] rb_posterior;
    {
        vector[N_GQ] rel_mean = relation_lin(vkick_GQ, a);
        for(i in 1:N_GQ){
            rb_posterior[i] = lower_trunc_normal_rng(rel_mean[i], err, 0.);
        }
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(rb[i] | relation_lin(vkick[i], a), err);
    }
}