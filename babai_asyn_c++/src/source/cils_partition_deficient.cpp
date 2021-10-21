
// Created by shilei on 10/6/21.
namespace cils {

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_partition_deficient(scalar *z, scalar *Q_tilde,
                                                        scalar *R_tilde, scalar *H_A, scalar *P_cum) {
        vector<scalar> H_P, I_K, P_tmp(n * n, 0), Piv, b_H, y, z_p;
        vector<index> x;
        index lastCol, t, b_i, i, k, nx, vlen;

        // 'partition_H_2:23' H_A = H;
        // for (i = 0; i < m * n; i++) {
        //     H_A[i] = H[i];
        // }
        // 'partition_H_2:24' z = z_B;
        // 'partition_H_2:25' lastCol = n;
        lastCol = n;
        // 'partition_H_2:26' P_cum = eye(n);
        helper::eye(n, P_cum);
        // 'partition_H_2:27' R_tilde = zeros(m,n);
        for (i = 0; i < m * n; i++) {
            R_tilde[i] = 0.0;
            Q_tilde[i] = 0.0;
        }
        // 'partition_H_2:29' indicator_tmp = zeros(2, n);
        // 'partition_H_2:30' i = 0;
        b_i = 0U;
        // 'partition_H_2:32' while lastCol >= 1
        while (lastCol >= 1.0) {
            index size_H, c_i, i1, i2, i3, r;
            // 'partition_H_2:33' firstCol = max(1, lastCol-m+1);
            t = std::fmax(1.0, (lastCol - m) + 1.0);
            // 'partition_H_2:34' H_cur = H_A(:, firstCol:lastCol);
            if (t > lastCol) {
                i = -1;
                i1 = -1;
            } else {
                i = t - 2;
                i1 = lastCol - 1;
            }
            size_H = i1 - i;
            // 'partition_H_2:35' z_cur = z(firstCol:lastCol);
            if (t > lastCol) {
                i2 = 1;
            } else {
                i2 = t;
            }
            // 'partition_H_2:36' P_tmp = eye(n);
            P_tmp.assign(n * n, 0);
            helper::eye(n, P_tmp.data());
            // Find the rank of H_cur
            // 'partition_H_2:39' [Q_qr,R_qr,P_qr]=qr(H_cur);
            b_H.resize(m * size_H);
            b_H.assign(m * size_H, 0);
            for (i3 = 0; i3 < size_H; i3++) {
                for (c_i = 0; c_i < m; c_i++) {
                    b_H[c_i + m * i3] = H_A[c_i + m * ((i + i3) + 1)];
                }
            }
            cils_reduction<scalar, index> reduction(m, size_H, 0, upper, 0, 0);
            reduction.cils_eml_qr(b_H.data());
            //  coder::eml_qr(b_H, reduction.Q, reduction.R_Q, reduction.P);
            // 'partition_H_2:41' if size(R_qr,2)>1
            if (size_H > 1) {
                // 'partition_H_2:42' r = sum( abs(diag(R_qr)) > 10^(-6) );
                vlen = size_H;
                if (m < vlen) {
                    vlen = m;
                }
                x.resize(vlen);
                x.assign(vlen, 0);
                for (k = 0; k < vlen; k++) {
                    x[k] = (std::abs(reduction.R_Q[k + m * k]) > 1.0E-6);
                }
                if (vlen == 0) {
                    r = 0;
                } else {
                    r = x[0];
                    for (k = 2; k <= vlen; k++) {
                        r += x[k - 1];
                    }
                }
            } else {
                // 'partition_H_2:43' else
                // 'partition_H_2:44' r = sum( abs(R_qr(1,1)) > 10^(-6));
                r = (std::abs(reduction.R_Q[0]) > 1.0E-6);
            }
            // 'partition_H_2:46' H_P = H_cur * P_qr;
            b_H.resize(m * size_H);
            b_H.assign(m * size_H, 0);
            H_P.resize(m * size_H);
            H_P.assign(m * size_H, 0);
            for (i3 = 0; i3 < size_H; i3++) {
                for (c_i = 0; c_i < m; c_i++) {
                    b_H[c_i + m * i3] = H_A[c_i + m * ((i + i3) + 1)];
                }
            }
            helper::mtimes_AP(m, size_H, b_H.data(), reduction.P.data(), H_P.data());
            // 'partition_H_2:47' z_p = P_qr' * z_cur;
            z_p.resize(size_H);
            z_p.assign(size_H, 0);
            for (k = 0; k < size_H; k++) {
                for (c_i = 0; c_i < size_H; c_i++) {
                    z_p[c_i] = z_p[c_i] + reduction.P[c_i * size_H + k] * z[(i2 + k) - 1];
                }
            }
            // 'partition_H_2:49' size_H_2 = size(H_cur, 2);
            // The case where H_cur is rank deficient
            // 'partition_H_2:52' if r < size_H_2
            if (r < size_H) {
                // Permute the columns of H_A and the entries of z
                // 'partition_H_2:54' H_A(:, firstCol:firstCol+size_H_2-1 -r ) = H_P(:,
                // r+1:size_H_2);
                if (r + 1 > size_H) {
                    i2 = 0;
                    i3 = -1;
                } else {
                    i2 = r;
                    i3 = size_H - 1;
                }
                index d = t + size_H;
                index d1 = (d - 1) - r;
                if (t > d1) {
                    c_i = 1;
                } else {
                    c_i = t;
                }
                nx = i3 - i2;
                for (i3 = 0; i3 <= nx; i3++) {
                    for (k = 0; k < m; k++) {
                        H_A[k + m * ((c_i + i3) - 1)] = H_P[k + m * (i2 + i3)];
                    }
                }
                // 'partition_H_2:55' H_A(:, firstCol+size_H_2-r: lastCol) = H_P(:, 1:r);
                if (1 > r) {
                    vlen = 0;
                } else {
                    vlen = r;
                }
                d -= r;
                if (d > lastCol) {
                    i2 = 1;
                } else {
                    i2 = d;
                }

                for (i3 = 0; i3 < vlen; i3++) {
                    for (c_i = 0; c_i < m; c_i++) {
                        H_A[c_i + m * ((i2 + i3) - 1)] = H_P[c_i + m * i3];
                    }
                }
                // 'partition_H_2:56' z(firstCol:firstCol+size_H_2-1-r) =
                // z_p(r+1:size_H_2);
                if (r + 1 > size_H) {
                    i2 = 0;
                    i3 = 0;
                } else {
                    i2 = r;
                    i3 = size_H;
                }
                if (t > d1) {
                    c_i = 1;
                } else {
                    c_i = t;
                }
                vlen = i3 - i2;
                for (i3 = 0; i3 < vlen; i3++) {
                    z[(c_i + i3) - 1] = z_p[i2 + i3];
                }
                // 'partition_H_2:57' z(firstCol+size_H_2-r: lastCol) = z_p(1:r);
                if (1 > r) {
                    vlen = 0;
                } else {
                    vlen = r;
                }
                d = (t + size_H) - r;
                if (d > lastCol) {
                    i2 = 1;
                } else {
                    i2 = d;
                }
                for (i3 = 0; i3 < vlen; i3++) {
                    z[(i2 + i3) - 1] = z_p[i3];
                }
                // Update the permutation matrix P_tmp
                // 'partition_H_2:60' I_K = eye(size_H_2);
                I_K.resize(size_H * size_H);
                I_K.assign(size_H * size_H, 0);
                helper::eye(size_H, I_K.data());
                // 'partition_H_2:61' Piv = eye(size_H_2);
                Piv.resize(size_H * size_H, 0);
                helper::eye(size_H, Piv.data());
                // 'partition_H_2:62' Piv(:, size_H_2-r+1:size_H_2) = I_K(:, 1:r);
                if (1 > r) {
                    vlen = 0;
                } else {
                    vlen = r;
                }
                d = (size_H - r) + 1;
                if (d > i1 - i) {
                    i = 1;
                } else {
                    i = d;
                }
                for (i1 = 0; i1 < vlen; i1++) {
                    for (i2 = 0; i2 < size_H; i2++) {
                        Piv[i2 + size_H * ((i + i1) - 1)] = I_K[i2 + size_H * i1];
                    }
                }
                // 'partition_H_2:63' Piv(:, 1:size_H_2-r) = I_K(:, r+1:size_H_2);
                if (r + 1 > size_H) {
                    i = 0;
                    size_H = 0;
                } else {
                    i = r;
                }
                nx = size_H - i;
                for (i1 = 0; i1 < nx; i1++) {
                    for (i2 = 0; i2 < size_H; i2++) {
                        Piv[i2 + size_H * i1] = I_K[i2 + size_H * (i + i1)];
                    }
                }
                // 'partition_H_2:64' P_tmp(firstCol:lastCol, firstCol:lastCol) = P_qr * Piv;
                if (t > lastCol) {
                    i = 1;
                    i1 = 1;
                } else {
                    i = t;
                    i1 = t;
                }

                I_K.assign(size_H * size_H, 0);
                helper::mtimes_AP(size_H, size_H, reduction.P.data(), Piv.data(), I_K.data());
                for (i2 = 0; i2 < size_H; i2++) {
                    for (i3 = 0; i3 < m; i3++) {
                        P_tmp[((i + i3) + n * ((i1 + i2) - 1)) - 1] = I_K[i3 + m * i2];
                    }
                }
            } else {
                // 'partition_H_2:65' else
                // Permute the columns of H_A and the entries of z
                // 'partition_H_2:67' H_A(:, firstCol:lastCol) = H_P;
                if (t > lastCol) {
                    i = 1;
                } else {
                    i = t;
                }
                for (i1 = 0; i1 < size_H; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        H_A[i2 + m * ((i + i1) - 1)] = H_P[i2 + m * i1];
                    }
                }
                // 'partition_H_2:68' z(firstCol:lastCol) = z_p;
                if (t > lastCol) {
                    i = -1;
                    i1 = 0;
                } else {
                    i = t - 2;
                    i1 = lastCol;
                }
                vlen = (i1 - i) - 1;
                for (i1 = 0; i1 < vlen; i1++) {
                    z[(i + i1) + 1] = z_p[i1];
                }
                // Update the permutation matrix P_tmp
                // 'partition_H_2:71' P_tmp(firstCol:lastCol, firstCol:lastCol) = P_qr;
                if (t > lastCol) {
                    i = 1;
                    i1 = 1;
                } else {
                    i = t;
                    i1 = t;
                }
                for (i2 = 0; i2 < size_H; i2++) {
                    for (i3 = 0; i3 < size_H; i3++) {
                        P_tmp[((i + i3) + n * ((i1 + i2) - 1)) - 1] = reduction.P[i3 + size_H * i2];
                    }
                }
            }
            // 'partition_H_2:73' P_cum = P_cum * P_tmp;
            I_K.resize(n * n);
            I_K.assign(n * n, 0);
            for (i = 0; i < n * n; i++) {
                I_K[i] = P_cum[i];
                P_cum[i] = 0;
            }

            helper::mtimes_AP(n, n, I_K.data(), P_tmp.data(), P_cum);

            // 'partition_H_2:75' firstCol = lastCol - r + 1;
            t = (lastCol - r) + 1.0;
            // 'partition_H_2:76' R_tilde(:, firstCol:lastCol) = R_qr(:, 1:r);
            if (1 > r) {
                vlen = 0;
            } else {
                vlen = r;
            }
            if (t > lastCol) {
                i = 1;
            } else {
                i = t;
            }
            for (i1 = 0; i1 < vlen; i1++) {
                for (i2 = 0; i2 < m; i2++) {
                    R_tilde[i2 + m * ((i + i1) - 1)] = reduction.R_Q[i2 + m * i1];
                }
            }
            // 'partition_H_2:77' Q_tilde(:, firstCol:lastCol) = Q_qr(:, 1:r);
            if (1 > r) {
                vlen = 0;
            } else {
                vlen = r;
            }
            if (t > lastCol) {
                i = 1;
            } else {
                i = t;
            }

            for (i1 = 0; i1 < vlen; i1++) {
                for (i2 = 0; i2 < m; i2++) {
                    Q_tilde[i2 + m * ((i + i1) - 1)] = reduction.Q[i2 + m * i1];
                }
            }
            // 'partition_H_2:79' i = i + 1;
            b_i++;
            lastCol -= r;
        }
        return {{}, 0, 0};
    }

} // namespace cils


