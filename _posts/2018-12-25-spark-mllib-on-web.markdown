---
layout:     post
title:      "Spark MLlib算法调用展示平台及其实现过程"
subtitle:   "介绍及代码实现"
date:       2018-12-25
author:     "fansy1990"
header-img: "img/post-bg-css.jpg"
catalog: true
tags:
  - Spark MLlib
  - 算法 
---

# 1. 软件版本：
 IDE：Intellij IDEA 14，
 Java：1.7，
 Scala：2.10.6；
 Tomcat：7，
 CDH：5.8.0； 
 Spark：1.6.0-cdh5.8.0-hadoop2.6.0-cdh5.8.0 ； 
 Hadoop：hadoop2.6.0-cdh5.8.0；(使用的是CDH提供的虚拟机)

# 2. 工程下载及部署：
 Scala封装Spark算法工程：https://github.com/fansy1990/Spark_MLlib_Algorithm_1.6.0.git ；
 调用Spark算法工程：https://github.com/fansy1990/Spark_MLlib_1.6.0_.git ；
 部署（主要针对Spark_MLlib_1.6.0工程）：
  1）配置好db.properties中相应用户名密码／数据库等参数；
  2）第一次启动tomcat，修改hibernate.cfg.xml文件中的hibernate.hbm2ddl.auto值为create，第二次启动修改为update；
  3) 打开集群参数页面，点击初始化，初始化集群参数，如果集群参数和当前集群不匹配，那么需要做相应修改；
   暂时考虑使用配置文件的方式来配置集群参数，如果要调整为数据库配置，那么修改Utisl.dbOrFile参数即可；即，暂时只需修改utisl.properties文件；
  4）拷贝Spark_MLlib_Algorithm_1.6.0工程生成的算法到到3）中spark.jar所在路径；
  5）拷贝集群中的yarn-site.xml到3）中spark.files所在路径；
  6）拷贝spark-assembly-1.6.0-cdh5.8.0-hadoop2.6.0-cdh5.8.0.jar到3）中spark.yarn.jar所在路径；

# 3. 工程实现原理：
## 3.1 Scala封装Spark算法工程：
  3.1.1 工程目录
   1. 工程目录如下所示：
![deploy](/img/blog/mllibonweb/path.png)
其中，data目录为所有的测试数据所在目录，这里针对不同的算法建立了不同的目录，主要有5类：分类与回归／聚类／协同过滤／降维／频繁项集挖掘；
main／scala里面就是所有封装Spark源码中的代码；
test／scala里面对应每个封装代码的测试；
  2. 工程采用Maven构建，直接根据pom文件加载对应依赖；
  3. 该工程需要经过maven打包，把打包好的jar包放到CDH的虚拟机中的HDFS上某一固定目录，方便Spark算法调用工程调用（具体目录下文有说）；
  
  3.1.2 单个算法实现（封装／测试），比如针对逻辑回归
  1. 针对逻辑回归，其封装代码如下所示：
  代码清单3-1 逻辑回归算法封装（Scala） 
   
```scala
package com.fz.classification

import com.fz.util.Utils
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 逻辑回归封装算法
 * Labels used in Logistic Regression should be {0, 1, ..., k - 1} for k classes multi-label classification problem
 * 输入参数：
 * testOrNot : 是否是测试，正常情况设置为false
 * input：输出数据；
 * minPartitions : 输入数据最小partition个数
 * output：输出路径
 * targetIndex：目标列所在下标，从1开始
 * splitter：数据分隔符；
 * method：使用逻辑回归算法："SGD" or "LBFGS"
 * hasIntercept : 是否具有截距
 * numClasses: 目标列类别个数；
 * Created by fanzhe on 2016/12/19.
 */
object LogisticRegression {

   def main (args: Array[String]) {
    if(args.length != 9){
      println("Usage: com.fz.classification.LogisticRegression testOrNot input minPartitions output targetIndex " +
        "splitter method hasIntercept numClasses")
      System.exit(-1)
    }
     val testOrNot = args(0).toBoolean // 是否是测试，sparkContext获取方式不一样, true 为test
     val input = args(1)
     val minPartitions = args(2).toInt
     val output = args(3)
     val targetIndex = args(4).toInt // 从1开始，不是从0开始要注意
     val splitter = args(5)
     val method = args(6) //should be "SGD" or "LBFGS"
     val hasIntercept = args(7).toBoolean
     val numClasses = args(8).toInt

     val sc =  Utils.getSparkContext(testOrNot,"Logistic Create Model")

     // construct data
     // Load and parse the data
     val training = Utils.getLabeledPointData(sc,input,minPartitions,splitter,targetIndex).cache()

     // Run training algorithm to build the model
     val model = method match {
       case "SGD" => new LogisticRegressionWithSGD()
         .setIntercept(hasIntercept)
         .run(training)
       case "LBFGS" => new LogisticRegressionWithLBFGS().setNumClasses(numClasses)
         .setIntercept(hasIntercept)
         .run(training)
       case _ => throw new RuntimeException("no method")
     }
     // save model

     model.save(sc,output)

     sc.stop()
  }
}
```

在上面的代码中，有对每个参数的解释，包括参数的含义，参数有哪些参数等；
在Main函数中，首先对各个参数进行获取并赋值变量，接着就是获取SparkContext；
其中，最重要的部分就是调用Spark自己封装的LogisticRegressionWithSGD 或 LogisticRegressionWithBFGS类进行逻辑回归建模；
最后，调用模型的save方法，把模型固化到HDFS上；
基本，所有的算法封装都采取这种模式，及对Spark MLlib中原生的算法再加一层封装。

 2. 测试
测试主要使用JUnit进行测试，其逻辑回归示例代码如下：
代码清单3-2 逻辑回归算法封装测试（Scala）

```scala
package com.fz.classification

import java.io.File

import com.fz.util.Utils
import org.junit.{Assert, Test}
import Assert._
/**
 * 测试Logistics Regression算法
 * Created by fanzhe on 2016/12/19.
 */
@Test
class LogisticRegressionTest {

  @Test
  def testMain1()={
//    testOrNot input output targetIndex splitter method hasIntercept numClasses
    val args = Array(
      "true",
      "./src/data/classification_regression/logistic.dat",
      "2",
      "./target/logistic/tmp1",
      "1",
      " ",
      "SGD",
      "true",
      "2" // this parameter is useless
    )
    // 删除输出目录
    Utils.deleteOutput(args(3))
    LogisticRegression.main(args)
    assertTrue(Utils.fileContainsClassName(args(3)+"/metadata/part-00000",
      "org.apache.spark.mllib.classification.LogisticRegressionModel"))
  }

  @Test
  def testMain2()={
    //    testOrNot input minPartitions output targetIndex splitter method hasIntercept numClasses
    val args = Array(
      "true",
      "./src/data/classification_regression/logistic.dat",
      "2",
      "./target/logistic/tmp2",
      "1",
      " ",
      "LBFGS",
      "true",
      "2"
    )
    // 删除输出目录
    Utils.deleteOutput(args(3))
    LogisticRegression.main(args)
    assertTrue(Utils.fileContainsClassName(args(3)+"/metadata/part-00000",
      "org.apache.spark.mllib.classification.LogisticRegressionModel"))
  }
}
```

这里面的方法都是第一步先构建算法参数；接着调用main方法；第三步，查看输出中是否具有模型的相关信息；
当然，这里面还可以添加多个测试方法，使用不同的算法参数或数据进行测试；（读者可自行添加）

## 3.2 Spark算法调用工程：
3.2.1 界面介绍
1. 首页：
![deploy](/img/blog/mllibonweb/home.png)
在系统首页有对该系统实现算法的介绍，系统主要功能有：
1）集群参数维护：主要是底层使用的Hadoop集群的参数配置，每次配置完成后，不仅仅会更新数据库对应记录，而且会更新Hadoop Configuration的获取；
2）监控：主要指Spark任务运行在YARN资源管理器下的任务状态监控；
3）文件上传及预览：文件上传主要是上传本地测试数据到HDFS上，方便页面进行测试；而预览则是查看HDFS上面的数据；
4）分类与回归／协同过滤／聚类／降维／关联规则：各个种类算法下面的每个算法的调用建模页面；
2. 集群参数页面：
![deploy](/img/blog/mllibonweb/param.png)
点击初始化，会把各个参数固定写入到后台数据库中，用户可以根据自己集群的配置不同，而进行参数修改，而每次修改也会刷新Hadoop 中Configuration的获取；
3. 监控：
![deploy](/img/blog/mllibonweb/monitor.png)
监控页面，会监控用户提交的SPark任务的运行状态，如果任务失败，则会显示异常信息（代码中只是截取了部分信息，需要进行调整，看如何可以得出重要的信息，直接显示）；后面会有具体实现过程分析。
4. 文件上传：
![deploy](/img/blog/mllibonweb/upload.png)
文件上传有两个功能:1)可指定一个本地目录和一个HDFS目录，然后把数据从本地上传到HDFS中；2）直接选择对应算法的数据，然后进行初始化，这个是把本地工程路径src/main/data中的对应数据上传到HDFS中的固定目录中；这两个上传的数据都可以在后面的算法建模中进行使用。
还有一点需要注意：被写入的HDFS路径是需要具有写权限的，而用户则是启动Tomcat的用户；
5. 文件查看：
![deploy](/img/blog/mllibonweb/view.png)
文件查看功能只能查看Text编码的文件，即文本文件，同时可以输入行号，即可进行文件内容的读取；
6. 逻辑回归算法：
![deploy](/img/blog/mllibonweb/logistics.png)
在逻辑回归算法界面，输入算法参数，点击提交，如果任务提交成功，即可在下面看到任务提交的ID，如果提交失败（即任务ID获取不到），同样有对应的提示信息；
同时，在任务提交后，在监控界面同样可以观察到该任务的状态，通过刷新即可获得最新的任务状态；

7. 其他算法与逻辑回归算法类似

3.2.2 架构
系统架构图如下所示（算法调用及监控）：
![deploy](/img/blog/mllibonweb/architecture.png)
流程描述如下：
1. 前台界面设置参数，包括算法数据、算法参数等，然后提交任务；
2. 任务提交后，CloudAction接收后，会发起一个线程，该线程会启动Hadoop上的一个Job，该Job有一个返回值，为任务ID，如果任务提交失败，则返回null；
3. 初级监控状态：CloudAction发起线程后，主线程阻塞，等待hadoop任务线程返回值，根据返回值状态，前台返回任务提交成功或失败；
4. 在3的同时，即可通过DBService来更新数据库相应表JobInfo的状态；
5. 在monitor.html界面，通过刷新按钮即可及时获取Hadoop任务状态（有相应的服务，见下文介绍），并更新数据库相关数据，返回前台所有任务信息；

## 3.2.3 部分实现细节
1. Spark提交任务
参考《基于Spark ALS在线推荐系统》；

2. monitor实时查询任务状态列表
monitor实时查询任务状态列表其流程描述如下：
1） 获取JobInfo中最新的records条记录；
2） 查找其中isFinished字段为false的数据；
3） 根据2）中查找的数据，去YARN获取其实时状态，并更新1）中的数据，然后存入数据库中；
4） 根据row和page字段分页返回JSON数据；
其代码如下所示：
代码清单3-3 更新监控任务列表

```scala
public void getJobInfo(){
        Map<String ,Object> jsonMap = new HashMap<String,Object>();
        // 1.
        List<Object> jobInfos = dBService.getLastNRows("JobInfo","jobId",true,records);
        // 2,3
        List<Object> list = null;
        try {
            list = HUtils.updateJobInfo(jobInfos);
            if(list != null || list.size()>0) {
                dBService.updateTableData(list);
            }
        }catch (Exception e){
            e.printStackTrace();
            log.warn("更新任务状态异常！");
            jsonMap.put("total", 0);
            jsonMap.put("rows", null);
            Utils.write2PrintWriter(JSON.toJSONString(jsonMap));
            return ;
        }
        // 4.
        jsonMap.put("total",list.size());
        jsonMap.put("rows",Utils.getSubList(list,page,rows));
        Utils.write2PrintWriter(JSON.toJSONString(jsonMap));
    }
 ```
 
第一步通过dBService获取给定records个记录；第二步则更新这些记录；看下HUtils.updateJobInfo的实现：
代码清单3-4 获取任务最新状态

```scala
public static List<Object> updateJobInfo(List<Object> jobInfos)throws YarnException,IOException{
        List<Object> list = new ArrayList<>();
        JobInfo jobInfo;
        for(Object o :jobInfos){
            jobInfo = (JobInfo) o;
            if(!jobInfo.isFinished()){ // 如果没有完成，则检查其最新状态
                ApplicationReport appReport=null;
                try {
                   appReport = getClient().getApplicationReport(SparkUtils.getAppId(jobInfo.getJobId()));
                } catch (YarnException  | IOException e) {
                    e.printStackTrace();
                    throw e;
                }
                /**
                 * NEW, 0
                 NEW_SAVING, 1
                 SUBMITTED, 2
                 ACCEPTED, 3
                 RUNNING, 4
                 FINISHED, 5
                 FAILED, 6
                 KILLED; 7
                 */
                switch (appReport.getYarnApplicationState().ordinal()){
                    case 0 | 1 | 2 |3 : // 都更新为Accepted状态
                        jobInfo.setRunState(JobState.ACCETPED);
                        break;
                    case 4 :
                        jobInfo.setRunState(JobState.RUNNING);break;
                    case 5:
//                        UNDEFINED,
//                                SUCCEEDED,
//                                FAILED,
//                                KILLED;
                        switch (appReport.getFinalApplicationStatus().ordinal()){
                            case 1: jobInfo.setRunState(JobState.SUCCESSED);
                            SparkUtils.cleanupStagingDir(jobInfo.getJobId());
                            jobInfo.setFinished(true);break;
                            case 2:
                                jobInfo.setRunState(JobState.FAILED);
                                SparkUtils.cleanupStagingDir(jobInfo.getJobId());
                                jobInfo.setErrorInfo(appReport.getDiagnostics().substring(0,Utils.EXCEPTIONMESSAGELENGTH));
                                jobInfo.setFinished(true);break;
                            case 3:
                                jobInfo.setRunState(JobState.KILLED);
                                SparkUtils.cleanupStagingDir(jobInfo.getJobId());
                                jobInfo.setFinished(true);break;
                            default: log.warn("App:" + jobInfo.getJobId() + "获取任务状态异常! " +
                                    "appReport.getFinalApplicationStatus():"+appReport.getFinalApplicationStatus().name()
                            +",ordinal:"+ appReport.getFinalApplicationStatus().ordinal());
                        }
                        break;
                    case 6:
                        jobInfo.setRunState(JobState.FAILED);
                        SparkUtils.cleanupStagingDir(jobInfo.getJobId());
                        jobInfo.setErrorInfo(appReport.getDiagnostics().substring(0,Utils.EXCEPTIONMESSAGELENGTH));
                        jobInfo.setFinished(true);break;
                    case 7:
                        jobInfo.setRunState(JobState.KILLED);
                        SparkUtils.cleanupStagingDir(jobInfo.getJobId());
                        jobInfo.setFinished(true);break;
                    default: log.warn("App:" + jobInfo.getJobId() + "获取任务状态异常!"+
                            "appReport.getYarnApplicationState():"+appReport.getYarnApplicationState().name()
                            +",ordinal:"+ appReport.getYarnApplicationState().ordinal());
                }
                jobInfo.setModifiedTime(new Date());
            }
            list.add(jobInfo);// 把更新后的或原始的JobInfo添加到list中
        }
 
        return list;
    }
```

这里的工作就是根据数据库中任务的状态，只查询任务没有完成的任务的最新状态，并更新原始任务状态，最后把更新后的或者原始任务添加到list中，并返回；
在代码清单3-3中，返回更新后的list后，接着调用了DBService.updateTableData,对数据进行固化；最后，使用subList对list进行截取，返回给前台某个分页的数据。

4. Spark算法调用工程后续开发：
不得不说，这个版本的工程还是没有开发完成的，那如果你想接着来开发，一般流程是怎样的呢？
。。。
1）编写src/main/java/下算法对应的Thread；
   2）编写webapp下的对应页面；
   3）编写webapp/js下对应的js；
   4）修改webapp/preprocess/upload.jsp，添加一条数据上传记录，并在main/data下添加对应的数据；
   5）启动工程，在页面上传数据，然后选择算法，设置参数，即可提交任务，提交任务后在监控界面即可看到算法运行状态；

工程状态（假设Scala工程为工程1，调用Spark算法工程为工程2）：
工程1：
基本封装了Spark Mllib中的数据挖掘相关算法，包括聚类、分类、回归、协同过滤、降维、频繁集挖掘（这个还有点问题）；
工程2：
目前只做了分类和回归算法的相关页面以及调用；

所以，如果你要在这个版本上开发，那么可以参考上面的流程先试着编写ALS算法的调用即可。

5. 总结
1. Spark算法调用工程还有很多页面没有完成，这个是类似重复性工作，并没有难点需要克服；
2. Spark算法调用工程中针对每个算法，本来是想在其算法调用界面加上其数据描述、算法描述、参数描述的，不过暂时还没有添加，but这些信息在Scala算法封装工程里面都有；
3. 关于使用SPARK ON YARN的方式调用Spark算法，并使用YARN来管理任务的流程基本在Spark算法调用工程中体现淋漓尽致了，再多也玩不出花儿了，所以如果有想学习研究这块内容的，则工程是一个很好的参考；
4. 之前对于分类算法这块是想加算法对比分析的，然后再加上些图表之类的展示，这样就显得更加高大上了，不过目前只进行了一步，就是写了个分类算法评估的Scala封装算法；
5. 可以考虑一些流程性的定时任务之类的加入到工程中，这样其实有点像Oozie了，不过为什么Oozie里面没有直接拖拽界面或流程任务监控管理的东西，如果有的话其实就更加像一个商业的软件了（Kettle）；
6. 关于SSH框架其实我是比较弱的，所以里面应用ssh的地方只是简单的应用（比如说在返回分页的时候，我直接用的是subList，这个应该是不妥的）；
7. 关于前台页面展现，我也是比较弱的，所以界面风格或单页的相关信息显示之类的，看着还不能做到赏心悦目；
8. The Code is free ，just enjoy！

分享，成长，快乐
脚踏实地，专注
转载请注明blog地址：http://blog.csdn.net/fansy1990
--------------------- 
作者：fansy1990 
来源：CSDN 
原文：https://blog.csdn.net/fansy1990/article/details/62238078 
版权声明：本文为博主原创文章，转载请附上博文链接！
