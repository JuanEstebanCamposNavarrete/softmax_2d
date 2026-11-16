# softmax_2d
Softmax algorithm implementation for 2D binary files.

***Implementacion **desde cero** de un kernel Softmaz para matrices 2d '[B, F]' (filas = batch, columnas = features), en **c++17 + cuda c++** , sin CUB, Thrust, cuBLAS ni ATen en el computo. Incluye perfilado con Nsight Compute, validacion con Compute Sanitizer y binding minimalista a PyTorch.***


Este proyecto ha sido desarrollado y probado con la siguiente configuración:

| Componente | Versión |
|------------|---------|
| **CUDA Toolkit** | 13.2 (V13.2.86) |
| **Compilador CUDA** | Build `cuda_13.2.r13.2/compiler.37953736_0` |
| **GCC** | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| **GPU** | NVIDIA GeForce GTX 1650 |
| **Sistema Operativo** | Ubuntu 24.04 |

---

## 1- Tabla de comentarios
-[Objetivo](#2-objetivo) \
-[Estructura del repositorio](#3-estructura-del-repositorio) \
-[Decisiones de diseño (lo que no se usa)](#4-decisiones-de-diseño) \
-[Evolucion del kernel(V0 - V1 - V2)](#5-evolucion-del-kernel) \
-[Requisitos](#6-requisitos) \
-[Uso](#7-uso) \
-[Metodologia de medicion](#8-metodologia-de-medicion) \
-[Perfilado con Nsight Compute](#9-perfilado-con-nsight-compute) \
-[Validacion con Compute Sanitizer](#10-validacion-con-compute-sanitizer) \
-[Resultados](#11-resultados) \
-[Correctitud numerica](#12-correctitud-numerica) \
-[Conclusiones](#13-conclusiones) 

---

## 2- Objetivo

Demostrar dominio practico de:

| Área | Evidencia |
|------|:---------:|
| Optimización de memoria | Coalescing, uso de `__ldg`, reducción warp-shuffle |
| Reducción a nivel de warp | `__shfl_xor_sync` manual (sin CUB) |
| Perfilado | `ncu` con métricas de occupancy, memory throughput, stalls |
| Correctitud numérica | Estabilidad con `max`, comparación vs. PyTorch |
| Aislamiento | Sin `cudaMallocManaged`, sin wrappers RAII |


## 3- Estructura del repositorio

```
softmax_2d/
├── binding
│   ├── binding.py
│   └── softmax_ext.cpp
├── CMakeLists.txt
├── data
│   ├── 1024x1024.bin
│   ├── 128x4096.bin
│   ├── 1x10000.bin
│   └── 4096x128.bin
├── include
│   ├── soft_kernel.cuh
│   ├── soft_utils.hpp
│   └── timing.hpp
├── README.md
├── scripts
│   └── gen.cpp
├── Sin título 1.docx
└── src
    ├── host_utils.cpp
    ├── main.cpp
    ├── soft_v0.cu
    ├── soft_v1.cu
    ├── soft_v2.cu
    └── timing.cu
```

---

## 4- Desiciones de diseño

Restricciones deliberadas:

| Restricción | Motivo |
|-------------|:------:|
|CUB / Thrust | Se implementa reducción manual con `__shfl_xor_sync` + memoria compartida. Demuestra comprensión del hardware warp. |
|cuBLAS / cuDNN / ATen | El binding a PyTorch **solo lanza** el kernel; el cómputo es 100% CUDA propio. |
|`cudaMallocManaged` | Se usa `cudaMalloc` + `cudaMemcpy` explícito para forzar control manual de transferencias. |
|`std::chrono` para kernels | Toda medición GPU usa `cudaEvent`. `std::chrono` solo se usa para I/O. |
|`cudaMallocAsync` / RAII | Sin wrappers; gestión explícita con `cudaFree` en cada path de salida. |
|Copiar tutoriales | El README documenta el proceso iterativo (V0 → V1 → V2). |

**Flags de compilacion:**
```
nvcc -03 -lineinfo -arch=sm_75
```
`-lineinfo` es indispensabel para que `ncu` asocie metricas a lineas funete.

---

## 5- Evolucion del kernel

>**V0 - Naive:**
>
>>1 hilo por elemento de salida. Sin memoria compartida. Sin vectorizacion.

```
-Codigo + metricas ncu (tabla).
-Cuello de botella identificado: lecturas redundantes de memoria global.
```
---
>**V1 - Reduccion con memoria compartida:**
>
>>Un bloque de hilos por fila.
>>Dos pasadas sobre la fila, pero cargando la fila a memoria compartida una sola vez.

```
-Explicacion de la reduccion jerarquica.
-Metricas ncu (tabla comparativa).
```
---

>**V2 - Vectorizacion y estabilidad numerica:**
> 
>>Sobre V1 agregar **estabilidad numerica** y **vectorizacion**.

```
--Explicacion de **float4** y resta del maximo.
Metricas ncu (tabla comparativa final).
```

## 6- Requisitos

-CUDA Toolkit ≥ 12.0
-GPU con arquitectura `smi_75` (para `__shfl_xor_sync` completp)
-PyTorch ≥ 2.0 (solo para el binding de test)

---

## 7-Uso

### Benchmark nativo
```bash
./build/softmax_bench --B 4096 --F 1024 --iters 100 --version v2
```
### Python

```Python
import torch, softmax
```

## 8- Metodologia de medicion

### faltante ###

Reglas:
1. **Warm-up obligatorio** (≥ 10 iteraciones).
2. Mismo buffer de entrada en todas las iteraciones (no cuenta H2D/D2H).
3. Reportar `media ± desviación` sobre ≥ 5 ejecuciones del benchmark.
4. `std::chrono` **nunca** envuelve lanzamientos de kernel.

## 9. Perfilado con Nsight Compute

### FALTANTE ###

## 10. Validación con Compute Sanitizer

### FALTANTE ###

## 11. Resultados

|Tabla: Versión | Tiempo | DRAM% | SM% | Speedup|
|---------------|--------|-------|-----|--------|
|**V0** | ****** | ***** | *** | ****** |
|**V1** | ****** | ***** | *** | ****** |
|**V2** | ****** | ***** | *** | ****** |

|Métrica (ncu)| Por qué importa |
|-----------|:-----------------:|
|gpu__time_duration.sum	|Tiempo de ejecución del kernel. Base para el speedup.|
|dram__throughput.avg.pct_of_peak_sustained_elapsed | Porcentaje del ancho de banda DRAM teórico alcanzado. Indica si estás limitado por memoria.|
|sm__throughput.avg.pct_of_peak_sustained_elapsed | Porcentaje de utilización de los SM. Indica si estás limitado por cómputo.|
|l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum | Bytes totales leídos de memoria global. Debe bajar (o mantenerse) al optimizar.|
|launch__occupancy_limit_registers | Si los registros limitan la ocupación.|
|launch__occupancy_limit_shared_mem	| Si la memoria compartida limita la ocupación.|
|smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct | Eficiencia de acceso coalescente. Debe acercarse al 100%.|
|smsp__inst_executed.sum | Instrucciones ejecutadas. Menos instrucciones suele indicar mejor uso de recursos.|

```
[Gráfico de barras simple]
```