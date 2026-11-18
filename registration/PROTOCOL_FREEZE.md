# PROTOCOL_FREEZE (copia del archivo que generó el notebook)

> **Origen:**
>
> - Copia textual de `config/PROTOCOL_FREEZE.md` en Drive (`MyDrive/CLM_losses/colab/`).
> - Lo generó la celda 8.1 de la rev. 4.4 (código v0.3.4, hash `94211b491b3f5d44`) el 2026-10-06 a las 02:33:41.
> - El JSON completo está en `config/PROTOCOL_FREEZE.json` [VERIFIED en Drive].
> - El investigador confirmó el freeze el 2026-10-05 (DP-06).
>
> **Nota del JSON:** Compuertas que no pasaron en F0: ['G2 reproducción InfoNCE', 'G3 atajos (modelo ciego)']. Enmienda A1 aprobada por el investigador el 2026-10-01 como respuesta a G2 y G3 (primaria T-mine, T-gen co-primaria, selección por val-mine, receta decidida en F0.7b). El 2026-10-02 pidió que 'Ejecutar todo' corra el estudio completo sin opciones; la única decisión pendiente es el freeze.
>
> **Adenda (C2c):** este freeze no registra C2c. Se registró aparte, sin modificar el freeze, en `config/PROTOCOL_FREEZE_ADDENDUM.json`, creado por la celda 8.1 de la rev. 4.6 el 2026-10-07 a las 01:52:08 UTC con la especificación de ENMIENDA_03 §14. Su texto está al final de este archivo. Motivo y desviación en ENMIENDA_03 §15 y DP-07; resultado en EXP-011.

---

# PROTOCOL FREEZE — 2026-10-06T02:33:41

code_hash: `94211b491b3f5d44` · fase de origen: F1b · receta: actual · selección: valmine_top1

Métrica primaria: top-1 en T-mine (test confirmatorio); co-primaria: macro de T-gen con los generadores del panel congelado, Holm entre ambas (ENMIENDA_03, A1.1)

C1: **vicreg_x** vs **infonce** (elegido en validación de F1b)

Panel T-gen: versión 1c0088495926 · construcción E (negativos editados (cambian sólo el dato clave) + oro original) · macro ['aya32b', 'glm4_32b', 'gptoss20b', 'mistral_small24b', 'phi4_14b']

Secundario pre-especificado: H1b (secundario pre-especificado; ENMIENDA_03 §12.4): por par (infonce_hn − infonce; siglip_hn − siglip), C2 = Δ(T-gem) − Δ(macro de T-gen) en los ítems del macro del test confirmatorio, bootstrap pareado por documento; H1b se sostiene en un par si los HN suben T-gem (Δ > +1.0 pp, IC95 > 0) y C2 > +1.0 pp con p_Holm < 0.05 (Holm entre los dos pares). No cambia C1. C2 mezcla generador y tipo de negativo (libre frente a editado); C2b los separa.

Secundario pre-especificado: Descomposición de C2 (secundario pre-especificado; ENMIENDA_03 §13): sets S, P y E de los mismos generadores del piloto, generados para los primeros ítems del panel del test confirmatorio (8.2c). Por par de C2: generador = Δ(T-gem) − Δ(S) y construcción = Δ(S) − Δ(E) en los ítems comunes, bootstrap pareado por documento, Holm entre los 4 contrastes. Un componente se confirma si es > +1.0 pp con p_Holm < 0.05, sólo si la brecha total Δ(T-gem) − Δ(E) es > +1.0 pp con IC95 > 0. El oro reescrito (Δ(P) − Δ(S)) y las proporciones son descriptivos.

| brazo | val_best | HP congelados |
|---|---|---|
| infonce | 0.919941782951355 | `{'lr_mult': 0.27726498213310763, 'weight_decay': 0.01, 'tau_init': 0.1}` |
| infonce_hn | 0.8981077075004578 | `{'lr_mult': 0.30590411317692706, 'weight_decay': 0.01, 'tau_init': 0.05}` |
| siglip | 0.8558952212333679 | `{'lr_mult': 0.6103037751646367, 'weight_decay': 0.0001, 'loss.t_init': 5.0, 'loss.b_init': -10.0}` |
| siglip_hn | 0.8802765607833862 | `{'lr_mult': 0.2526369434719297, 'weight_decay': 0.0001, 'loss.t_init': 10.0, 'loss.b_init': -10.0}` |
| triplet_hn | 0.9057496190071106 | `{'lr_mult': 0.28611279239969867, 'weight_decay': 0.01, 'loss.margin': 0.08192186333536983}` |
| simsiam_x | 0.21106259524822235 | `{'lr_mult': 0.8610889055267814, 'weight_decay': 0.0, 'loss.pred_hidden': 128}` |
| byol_x | 0.23362445831298828 | `{'lr_mult': 3.6548137508039904, 'weight_decay': 0.0001, 'loss.tau_base': 0.999, 'loss.pred_hidden': 2048}` |
| barlow_x | 0.8857350945472717 | `{'lr_mult': 0.5765110088021451, 'weight_decay': 0.0, 'loss.exp_dim': 4096, 'loss.lambd': 0.0038384212380068363}` |
| vicreg_x | 0.8933770060539246 | `{'lr_mult': 0.4586698739156817, 'weight_decay': 0.0, 'loss.exp_dim': 2048, 'loss.sim': 12.241523081658814, 'loss.std': 16.762255278294464, 'loss.cov': 0.7207431164885922}` |
| mse_sigreg | 0.6706695556640625 | `{'lr_mult': 0.35026851986397295, 'weight_decay': 0.0, 'loss.lam': 0.005170959081843542, 'loss.n_slices': 1024}` |
| infonce_sigreg | 0.9090247750282288 | `{'lr_mult': 0.3068158422277282, 'weight_decay': 0.0001, 'tau_init': 0.07, 'loss.lam': 0.00013823415304432167}` |
| align_unif | 0.8384279608726501 | `{'lr_mult': 1.1910366762161104, 'weight_decay': 0.0, 'loss.gamma': 1.6539055432613214}` |
| infonce_lse | 0.9235807657241821 | `{'lr_mult': 0.5157356755193137, 'weight_decay': 0.0, 'tau_init': 0.07, 'loss.lam': 0.004384465260481652}` |
| dino_x | 0.17831149697303772 | `{'lr_mult': 1.1758201220684723, 'weight_decay': 0.01, 'loss.K': 4096, 'loss.tau_t_final': 0.07, 'loss.m_base': 0.99}` |
| swav_x | 0.7048762440681458 | `{'lr_mult': 0.4374759700362126, 'weight_decay': 0.0001, 'loss.K': 8192, 'loss.eps': 0.03, 'loss.tau': 0.2, 'loss.freeze_steps': 100}` |

---

# ADENDA AL PROTOCOL FREEZE — 2026-10-07T01:52:08 (copia del JSON en Drive)

Freeze: 2026-10-06T02:33:41 · código congelado `94211b491b3f5d44` · código de la adenda `e160594adcd46368` · registro: ENMIENDA_03 §14 (2026-10-05)

**Motivo.** El PROTOCOL FREEZE se creó el 2026-10-06T02:33:41 con el código 94211b491b3f5d44 (notebook rev. 4.4), que todavía no registraba C2c. C2c quedó especificado en ENMIENDA_03 §14 el 2026-10-05, antes del freeze y antes de abrir el test confirmatorio, y el investigador confirmó el freeze sabiendo que lo incluía. Esta adenda lo registra sin modificar el freeze, antes de entrenar la variante.

**Desviación.** C2c queda registrado fuera del JSON del freeze y se ejecuta después de ver los resultados de F3 de los 15 brazos. Su especificación (variante, pruebas, umbrales y Holm) es la de ENMIENDA_03 §14, sin cambios. Al reportarlo hay que declarar esta desviación.

Secundario pre-especificado: Mecanismo de H1b (secundario pre-especificado; ENMIENDA_03 §14): en F3 se entrena además la variante 'HN solos' de infonce_hn (F3_variants: mismos HP congelados y semillas; sólo el denominador pasa a hn_only) y se compara con infonce_hn ('HN + in-batch'), bootstrap pareado por documento. Prueba 1: top-1 en T-gem con estados barajados, se confirma si Δ > +1.0 pp con p_Holm < 0.05 (aprende una preferencia que no depende del estado). Prueba 2: top-1 en T-mine, se confirma si Δ < −1.0 pp con p_Holm < 0.05 (pierde discriminación real). Holm entre las 2. Control, fuera de Holm: infonce_hn − infonce en T-gem con estados barajados, equivalente por TOST ±1.0 pp (con in-batch, los HN no crean esa preferencia). T-gem, T-rand, T-mine-50 y el macro de T-gen son descriptivos.

Variantes de F3 (no entran al todos contra todos ni a C1): `infonce_hn_hnonly` = HP congelados de infonce_hn + `{"loss_params": {"hn_denominator": "hn_only"}}`
