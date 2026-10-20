Crypto Momentum Strategy Backtester (XGBoost)Este repositorio contiene un motor de Backtesting diseñado para evaluar estrategias de trading algorítmico en criptomonedas 
utilizando modelos de Machine Learning (XGBoost).El script simula operaciones históricas basadas en predicciones de probabilidad, gestionando capital, apalancamiento, 
comisiones y reglas estrictas de gestión de riesgo (Stop Loss, Take Profit y Time Stop).


Integración con modelos XGBoost pre-entrenados(.json).
Gestión de Riesgo: Lógica configurable para Stop Loss (SL) y Take Profit (TP).Time Stop: Cierre forzado de posiciones si el precio no se mueve a favor 
en un tiempo determinado (evita costes de oportunidad).Simulación Realista:Incluye comisiones de Exchange (Taker fees).Simula apalancamiento (Leverage).
Cálculo de interés compuesto (reinversión de capital).
Visualización: Genera una curva de equidad (Equity Curve) con matplotlib.

Python 3.8+ 
y las siguientes librerías:Bashpip install pandas numpy matplotlib xgboost scikit-learn


main.py:El script de backtesting (el código proporcionado).
model.py: Módulo auxiliar que debe contener las funciones data (carga de csv) y add_indicators (ingeniería de características).
crypto_momentum_25tp.json: El modelo XGBoost entrenado.
datos_btc_1año.csv: Dataset histórico con datos OHLCV.


Configuración de la EstrategiaLos parámetros clave de la estrategia se encuentran dentro de la función run_backtest.
Puedes ajustarlos según tu perfil de riesgo:
ParámetroValor por DefectoDescripciónInitial 
Capital 1000 USDTCapital inicial de la cuenta.
Leverage5xNivel de apalancamiento utilizado.
Threshold0.60 (60%)Probabilidad mínima que el modelo debe predecir para abrir un Long.
TP_PCT0.015 (1.5%)Objetivo de ganancia por operación (movimiento del precio sin apalancar).
SL_PCT0.005 (0.7%)Límite de pérdida por operación.
MAX_HOLD_CANDLES32Número máximo de velas antes de cerrar la posición (Time Stop).
Fee0.05%Comisión por operación (ej. Binance Taker).


Define las características (features) que usaste para entrenar el modelo en la sección final del script.
Ejecuta el script:Python# Ejemplo de ejecución dentro del script
df_test = data("datos_btc_1año.csv")
df_test = add_indicators(df_test)
model = obtener_modelo_entrenado()

# Asegúrate de que estas columnas coincidan con el entrenamiento
feature_cols = ["day", "hour", "volume"] 

run_backtest(df_test, model, feature_cols, leverage=5)
Resultados y MétricasAl finalizar la ejecución, el script imprimirá un resumen detallado en la consola:
Win Rate Real: Porcentaje de operaciones ganadoras.
Retorno Neto: Rendimiento total de la cuenta en %.
Max Drawdown: La mayor caída de capital desde un máximo histórico (medida de riesgo).
Desglose de Salidas: Cuántas operaciones cerraron por TP, SL o Time Stop.
Además, se abrirá una ventana con el gráfico de la Curva de Crecimiento de la Cuenta.
