package day11;
import javax.swing.*;
import java.awt.*;

public class MyApp extends JFrame {
	JPanel p;
	JPanel pN;
	JButton bt1, bt2, bt3, bt4; //pN에 붙이기
	
	JTextArea ta1, ta2; //p에 붙임. JScrollPane에 담아서 "Center"
	
	public MyApp() {
		super("::MyApp::");
		p=new JPanel();
		pN=new JPanel();
		add(pN);
		
		bt1=new JButton("Login");
		bt2=new JButton("Join");
		bt3=new JButton("Home");
		bt4=new JButton("Exit");
		
		pN.add(bt1, "North");
		pN.add(bt2, "North");
		pN.add(bt3,"North");
		pN.add(bt4, "North");
		
		ta1=new JTextArea(20,20);
		JScrollPane sp=new JScrollPane(ta1);
		pN.add(sp);
		
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		MyApp my=new MyApp();
		my.setSize(500,500);
		my.setVisible(true);
		
	}

}
