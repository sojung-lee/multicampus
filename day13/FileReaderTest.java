package day13;
/* FileReader/FileWriter
 * - 2byte(문자) 기반 스트림
 * - 노드 스트림 (파일과 노드 연결)
 * */
import java.io.*;
public class FileReaderTest {

	public static void main(String[] args) 
	throws IOException
	{
		String fname="C:\\Java\\Workspace\\Begin\\src\\day13\\FileInputStreamTest.java";
		//String fname="C:/myjava/readme.txt";
		
		File file=new File(fname);
		File file2=new File("C:/myjava/readme_copy2.txt");
		
		long fsize=file.length();//파일의 크기를 반환한다.
		System.out.println("파일의 크기: "+fsize+"bytes");
		
		FileReader fr=new FileReader(file);//노드 연결
		
		FileWriter fw=new FileWriter(file2, true);//true를 주면 기존 파일내용을 덮어쓰지 않고 덭붙여쓰기를 한다
		int n=0;
		char[] data=new char[1000];//달걀판
		while((n=fr.read(data))!=-1) {
			fw.write(data, 0,n);
			fw.flush();
			//System.out.write(n);
			//System.out.flush();
		}
		fr.close();
		

	}

}








