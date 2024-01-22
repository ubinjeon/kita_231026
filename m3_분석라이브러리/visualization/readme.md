# 시각화1
## 판다스 내장 그래프 도구
* 판다스는 Matplotlib 라이브러리의 기능을 일부 내장하고 있어서 별도로 임포드하지 않고 간단히 그래프 그릴 수 있음.
* 선그래프 : df.plot()메소드 적용시 다른 옵션을 추가하지 않으면 가장 기본적인 선 그래프를 그림.
* 막대 그래프 : df.plot(kind='bar'), df.plot(kind='barh',stacked=True,alpha=0.8)
* 히스토그램 : df.plot(kind='hist')
* 산점도 : df.plot(x=' ', y=' ', kind='scatter'), 상관관계를 볼 수 있음.
* 박스 플롯 : df.plot(kind='box'), df.boxplot(column=['feature']) 특정 변수의 데이터 분포와 분산 정도에 대한 정보를 제공.

## matplotlib
- histogram : plt.hist(x, bins= )
- line plot : plt.plot(x,y)
- plt.bar(x,y, width= , color='')
- scatter plot : plt.scatter(x, y):

## seaborn
- sns.distplot(data, bins=, color=, kde=), histplot, displot으로 변경
- sns.boxplot(x, y, data, color)
- sns.violinplot(x, y, data, color)
- sns.barplot(x, y, data, color)
- sns.jointplot(x, y, data, color)
- sns.pairplot(data, hue, palette)
- sns.lmplot(x,y,data,color) <- 선형회귀를 그림으로
- sns.regplot(x,y,data,fig_reg=False) <- 회귀를 그림으로

## np.random
- np.random.normal(loc,scale,size)은 더 일반적인 경우에 사용. 사용자가 평균과 표준 편차를 직접 지정할 수 있기 때문에, 특정한 정규 분포에서 샘플링이 필요한 경우에 적합
- np.random.randn은 표준 정규 분포(즉, 평균이 0이고 표준 편차가 1인 정규 분포)에서 샘플링할 때 사용. 이는 많은 통계적 실험과 시뮬레이션에서 기본값으로 쓰이는 분포.

## 코드예제
- df = pd.read_csv('/content/drive/MyDrive/KITA_1026/m3_분석라이브러리/visualization/dataset/주가데이터.csv')
- df['Ndate'] = pd.to_datetime(df['Date'])
- df1 = df.set_index('Ndate')
- df1.drop(['Date','Volume'],axis=1,inplace=True)
- df1.Close[::-1].plot(kind='bar')
- df1.Close.plot(kind='hist')
- 이상치 구하는 법
Q1=df.Start.describe()['25%']
Q3=df.Start.describe()['75%']
IQR = Q3-Q1
outliers = df[(df.Start > (Q3 + 1.5*IQR)) | (df.Start <(Q1-1.5*IQR))]
- subplot 그리는법
fig,axes = plt.subplots(2,2,figsize=(10,6))
axes[0,0].plot(age,height,'r--')
axes[0,1].hist(height,bins=10)
axes[1,0].scatter(age,height)
axes[1,1].bar(age,height)
- sin 그래프 예제
x = np.arange(0,4*np.pi,0.1)   # start,stop,step
y = np.sin(x)
plt.plot(x,y)
plt.title('sea level')
plt.ylabel('height')
plt.xlabel('time')
- heatmap 예제
df_flights = sns.load_dataset('flights')
df_flights.head()
pivot_flights = df_flights.pivot('year','month',"passengers")
plt.figure(figsize=(10,8))
sns.heatmap(pivot_flights, annot=True, fmt="d", cmap='plasma')
plt.title('Monthly Passengers by year')
plt.show()
- pairplot 예제
df_penguins = sns.load_dataset('penguins')
sns.pairplot(df_penguins, hue='species', palette='rainbow')
- boxplot, violinplot 예제
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
sns.boxplot(x = ttn['pclass'], y = ttn['age'], color = 'darkgray')
plt.subplot(1,2,2)
sns.violinplot(x = ttn['pclass'], y = ttn['age'], color = 'gray')
- countplot 예제
plt.figure(figsize=(10, 6))
sns.countplot(x='island', hue='species', data=penguins)
plt.title('Penguin Count by Island and Species')
plt.show()
