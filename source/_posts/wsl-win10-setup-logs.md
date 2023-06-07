---
title: wsl-win10-setup-logs
date: 2023-05-31 00:29:16
tags:
- WSL Usages
categories:
- WSL setup logs
---

# 本篇博客主要记录本人在windows10系统中安装WSL（用于windows的linux子系统）的详细过程

<<<<<<< HEAD
## coming soon
=======
## WSL是什么？*--from chatGPT*
WSL（Windows Subsystem for Linux）是一种在 Windows 操作系统上运行 Linux 环境的兼容层。它允许用户在 Windows 上使用原生的 Linux shell 和命令行工具，以及在 Linux 上运行的应用程序和工具。WSL 提供了一个完整的 Linux 内核接口，可以在 Windows 系统上运行 Linux 发行版，如 Ubuntu、Debian、Fedora 等。

WSL 的设计目标是提供一个无需双启动、虚拟机或容器的方式，在 Windows 上进行 Linux 开发和运行 Linux 应用程序。通过 WSL，开发人员可以利用 Windows 的生态系统和工具，并使用 Linux 的开发环境和工具链。WSL 提供了与 Linux 完全兼容的系统调用，可以在 Windows 上运行许多 Linux 软件包和应用程序。

WSL 分为两个主要版本：WSL 1 和 WSL 2。WSL 1 是通过将 Windows 和 Linux 之间的系统调用转换为 Windows API 实现的，而 WSL 2 则基于 Hyper-V 虚拟化技术，在 Windows 中运行一个完整的 Linux 内核。

使用 WSL，用户可以在 Windows 上进行各种任务，如开发和调试应用程序、运行脚本、使用命令行工具等，同时享受到 Windows 操作系统的优势和便利性。


## 安装WSL
1. 首先，保证windows10原生，不然后续会有很多错误出现；
2. 在微软商店安装WSL之前，请先设置Windows10系统中两个选项，打开“控制面板\程序\程序和功能\启用或关闭windows功能”，在弹出的窗口中勾选“适用于Linux的Windows子系统”、“虚拟机平台”两个选项；
3. 然后，为了保证WSL默认安装version=2，使用管理员身份打开windows10中的PowerShell，然后运行如下命令：
```
wsl --set-default-version 2
```
5. 此时，前序工作已经完成。然后，在Microsoft Store中搜索Ubuntu（建议下载18.04版本，本教程基于Ubuntu-18.04安装），并安装（默认安装位置在C:\\Windows\\System32下面，后边会介绍将WSL镜像迁移到其他系统盘）；
6. 使用管理员身份打开cmd/PowerShell，在命令行输入：
```
wsl -l -v
```
查看当前WSL状态，如果看到如下所示，状态为stopped，那么继续进行下一步操作：
```
  NAME            STATE           VERSION
* Ubuntu-18.04    Stopped         2
```
如果看到状态为：
```
  NAME            STATE           VERSION
* Ubuntu-18.04    Running         2
```
那么，请在窗口运行如下命令将状态更改为Stopped；
```
wsl --shutdown
```
## 将WSL迁移到其他系统盘（不放在C盘），并重新装载
1. 经过上述步骤之后，目前Ubuntu-18.04镜像已经被shutdown，同时存在于C盘下，然后使用管理员身份打开cmd/PowerShell，然后运行如下命令：
```
wsl --export Ubuntu-18.04 F:\ubuntu-18.04.tar
```
接着注销Ubuntu-18.04：
```
wsl --unregister Ubuntu-18.04
```
然后，从路径F:\ubuntu-18.04.tar下重新导入镜像：
```
wsl --import Ubuntu-18.04 F:\Ubuntu-18.04\ F:\ubuntu-18.04.tar --version 2
```

## 配置WSL
1. 使用管理员身份打开PowerShell，然后按照如下步骤进行；
2. 首先检查WSL是否装载成功
```
wsl -l -v
```
3. 然后，运行如下命令进入到WSL中：
```
wsl
```
4. 接着会让你设置root密码，随便设置即可
5. 后续操作同在Linux中一样，创建用户/配置python开发环境/配置C++开发环境/配置java开发环境（哈哈java开发还是推荐windows10+IDEA，太爽了）

## WSL容量扩充[1]（我个人理解相当于，将WSL系统能够使用的虚拟磁盘大小进行扩充，推荐1T，当然这取决于你自己主机的硬盘大小）
1. 首先，运行如下命令关闭Ubuntu-18.04：
```
wsl --shutdown
wsl -l -v （检查是否关闭）
```
2. 然后，运行如下命令查看wsl默认分配的空间大小（默认分配了256GB）：
```
wsl df -h /.
```
3. 进入到wsl镜像安装目录: F:\Ubuntu-18.04\ext4.vhdx
4. 然后，使用管理员身份打开PowerShell，运行如下命令以扩充WSL所依赖的虚拟磁盘的最大大小（也就是最大容量）
```
> diskpart
> Select vdisk file="F:\Ubuntu-18.04\ext4.vhdx"
> expand vdisk maximum=xxxxxxxx //MB为单位
```

5. 然后启动wsl，进入wsl，并运行如下命令[1]：
```
$ sudo mount -t devtmpfs none /dev
mount: /dev: none already mounted on /mnt/wsl.
$ mount | grep ext4
/dev/sdb on / type ext4 (rw,relatime,discard,errors=remount-ro,data=ordered)
$ sudo resize2fs /dev/sdb //这里是sdb sdc还是其他，取决于你自己的电脑
```

*至此，WSL Ubuntu-18.04安装完成，现在就可以以管理员身份打开PowerShell，运行wsl进入Linux进行体验了，后续还会陆续讲解如何使用windows10上vscode来连接wsl进行代码调试以及python/cpp/npm/nvm/nodejs等环境的搭建，敬请期待。*

**参考文章**
[1] https://blog.csdn.net/StarRain2016/article/details/122803337
>>>>>>> 0faf98f2fe074769adf459ed307fb014a35a9876
