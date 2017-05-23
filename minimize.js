function gradient(f,x) {
    var dim = x.length, f1 = f(x);
    if(isNaN(f1)) throw new Error('The gradient at ['+x.join(' ')+'] is NaN!');
    var tempX = x.slice(0), grad = Array(dim);
    for(var i=0; i<dim; i++) {
        var delta = Math.max(1e-6*f1, 1e-8);
        for (var k = 0;; k++) {
            if(k == 20) throw new Error("Gradient failed at index "+i+" of ["+x.join(' ')+"]");
            tempX[i] = x[i]+delta;
            var f0 = f(tempX);
            tempX[i] = x[i]-delta;
            var f2 = f(tempX);
            tempX[i] = x[i];
            if(!(isNaN(f0) || isNaN(f2))) {
                grad[i] = (f0-f2)/(2*delta)
                var t0 = x[i]-delta;
                var t1 = x[i];
                var t2 = x[i]+delta;
                var d1 = (f0-f1)/delta;
                var d2 = (f1-f2)/delta;
                var err = Math.min(Math.max(Math.abs(d1-grad[i]),Math.abs(d2-grad[i]),Math.abs(d1-d2)),delta);
                var normalize = Math.max(Math.abs(grad[i]),Math.abs(f0),Math.abs(f1),Math.abs(f2),Math.abs(t0),Math.abs(t1),Math.abs(t2),1e-8);
                if(err/normalize < 1e-3) break; //break if this index is done
            }
            delta /= 16
        }
    }
    return grad;
}


function numeric_norm2(x){
    for (var i = x.length - 1, ss = 0; i >= 0; i--) {
        ss += x[i] * x[i]
    }
    return Math.sqrt(ss)
}

function numeric_identity(n){
    return Array.from(Array(n), (_,i) => Array.from(Array(n), (_,j) => +(i==j)))
}

function numeric_inplace_negate(x){
    for (var i = x.length - 1; i >= 0; i--) {
        x[i] = -x[i]
    }
    return x;
}

function numeric_dot(a,b){
    if (Array.isArray(a[0])){
        return a.map(x=>numeric_dot(x,b))
    }
    return a.reduce((x,y,i) => x+y*b[i],0)
}

function numeric_sub(a,b){
    if(Array.isArray(a[0])){
        return a.map((c,i)=>numeric_sub(c,b[i]))
    }
    return a.map((c,i)=>c-b[i])
}

function numeric_add(a,b){
    if(Array.isArray(a[0])){
        return a.map((c,i)=>numeric_add(c,b[i]))
    }
    return a.map((c,i)=>c+b[i])
}

function numeric_div(a,b){
    return a.map(c=>c.map(d=>d/b))
}

function numeric_mul(a,b){
    if(Array.isArray(a[0])){
        return a.map(c=>numeric_mul(c,b))
    }
    return a.map(c=>c*b)
}

function numeric_tensor(a,b){
    return a.map((c,i)=>numeric_mul(b,c))
}

function minimize(f,x0,tol=1e-8,maxit=1000) {
    tol = Math.max(tol,2e-16);
    x0 = x0.slice(0);
    var n = x0.length;
    var f0 = f(x0);
    if(isNaN(f0)) throw new Error('minimize: f(x0) is a NaN!');
    var grad = a => gradient(f,a)
    var H1 = numeric_identity(n), g0 = grad(x0);
    for(var it = 0; it<maxit; it++) {
        if(!g0.every(isFinite)) { var msg = "Gradient has Infinity or NaN"; break; }
        var step = numeric_inplace_negate(numeric_dot(H1,g0));
        if(!step.every(isFinite)) { var msg = "Search direction has Infinity or NaN"; break; }
        var nstep = numeric_norm2(step);
        if(nstep < tol) { var msg="Newton step smaller than tol"; break; }
        var t = 1;
        var df0 = numeric_dot(g0,step);
        // line search
        var x1 = x0;
        var s;
        for(;it < maxit && t*nstep >= tol; it++) {
            s = numeric_mul(step,t);
            x1 = numeric_add(x0,s);
            var f1 = f(x1);
            if(!(f1-f0 >= 0.1*t*df0 || isNaN(f1))) break;
            t *= 0.5;
        }
        if(t*nstep < tol) { var msg = "Line search step size smaller than tol"; break; }
        if(it === maxit) { var msg = "maxit reached during line search"; break; }
        var g1 = grad(x1);
        var y = numeric_sub(g1,g0);
        var ys = numeric_dot(y,s);
        var Hy = numeric_dot(H1,y);
        H1 = numeric_sub(numeric_add(H1,numeric_mul(numeric_tensor(s,s),(ys+numeric_dot(y,Hy))/(ys*ys))),numeric_div(numeric_add(numeric_tensor(Hy,s),numeric_tensor(s,Hy)),ys));
        x0 = x1;
        f0 = f1;
        g0 = g1;
    }
    return {solution: x0, f: f0, gradient: g0, invHessian: H1, iterations:it, message: msg};
}
