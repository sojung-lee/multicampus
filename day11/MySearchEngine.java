package day11;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class MySearchEngine extends JFrame {

	JPanel p=new JPanel();
	public MySearchEngine() {
		super("::MySearchEngine::");
		add(p,"Center");
		
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		MySearchEngine my=new MySearchEngine();
		my.setSize(500,500);
		my.setVisible(true);
		
	}

}
